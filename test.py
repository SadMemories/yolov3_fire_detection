import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         imgsz=416,
         conf_thres=0.001,
         iou_thres=0.6,  # for nms
         save_json=False,
         single_cls=False,
         augment=False,
         model=None,
         dataloader=None,
         multi_label=True):
    # Initialize/load model and set device
    if model is None:
        is_training = False
        device = torch_utils.select_device(opt.device, batch_size=batch_size)  # opt.device = 0或1
        verbose = opt.task == 'test'  # verbose的含义暂时未知

        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg, imgsz)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)

        # Fuse
        # 见：https://zhuanlan.zhihu.com/p/110552861
        model.fuse()  # 将BN层和Conv层进行融合
        model.to(device)

        if device.type != 'cpu' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:  # called by train.py
        is_training = True
        device = next(model.parameters()).device  # get model device cuda:0
        verbose = False

    # Configure run
    data = parse_data_cfg(data)  # data/fire.data
    nc = 1 if single_cls else int(data['classes'])  # number of classes
    path = data['valid']  # path to test images data/test.txt
    names = load_classes(data['names'])  # class names data/fire.names names是列表的形式 ['fire']
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95 0.45/9 = 0.05
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95 iouv[0]返回的是一个数 .view(1)将其转变为数组 tensor([0.5])
    niou = iouv.numel()  # iouv.numel()=1

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, imgsz, batch_size, rect=True, single_cls=opt.single_cls, pad=0.5)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    _ = model(torch.zeros((1, 3, imgsz, imgsz), device=device)) if device.type != 'cpu' else None  # run once (含义暂时未知) 3
    coco91class = coco80_to_coco91_class()  # 含义未知
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)  # GIOU loss, obj loss, cls loss
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)  # targets = {Tensor:8*6} 第一个8表示的是gtbox的个数 6表示bs cls x y w h (xywh都是归一化之后的)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()
            '''
            在train.py调用test.py的时候inf_out train_out如下：
            train_out: {tuple:3} 每一个tuple是[bs, 3, grid, grid, 6]
            inf_out: {Tensor:3} torch.size([4, 10080, 6])
            '''
            inf_out, train_out = model(imgs, augment=augment)  # inference and training outputs 在train中augment是false
            # print(train_out)
            # print(train_out.shape)
            # print('---------------------')
            # print(inf_out)
            # print(inf_out.shape)
            t0 += torch_utils.time_synchronized() - t

            # Compute loss
            if is_training:  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = torch_utils.time_synchronized()
            # output是长度为batch-size的list，list中的每个元素是size为[k, 6]的tensor k表示经过NMS后box的个数
            # 6：前4表示x1 y1 x2 y2坐标 第5个表示conf 第6个表示种类
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=multi_label)
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):  # 对一个batch中的每张图片分别处理
            # targets[:, 0]表示的是batch中的第几张图片 1:对应的是cls x y w h
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)  # 每一张图片中gtbox的个数
            tcls = labels[:, 0].tolist() if nl else []  # target class 长度为nl 对应的种类下标
            seen += 1  # 作用未知

            if pred is None:  # pred为0 表示预测的该张图片中没有物体box
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            # 规范划pred中的坐标值
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:  # 默认为false
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            # pred.shape[0]表示预测出的box的个数 niou=1
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:  # nl表示真正的gtbox的个数
                detected = []  # target indices
                tcls_tensor = labels[:, 0]  # labels[:, 0]表示的cls

                # target boxes
                # 去归一化之后的x1 y1 x2 y2
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                # 对于火焰检测来说cls只能是0
                for cls in torch.unique(tcls_tensor):
                    # nonzero()返回tcls_tensor中非0的下标 tcls_tensor表示正确的种类
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # target indices
                    # pred[:,5]表示预测的种类
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # 固定下种类之后 将预测出来该种类的box和gtbox进行iou计算
                        # 返回的是每一个预测的box与哪一个gtbox的iou值最大 将每一个预测的box对应的iou最大值本身以及最大值的下标返回
                        # i对应的ti中的下标
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        # iouv[0]=0.5
                        # 返回ious中大于iouv[0]的下标
                        for j in (ious > iouv[0]).nonzero():
                            # ti[i[j]]表示tcls_tensor中的下标
                            d = ti[i[j]]  # detected target d对应的是tcls_tensor中的下标 表示第几个gtbox
                            if d not in detected:
                                detected.append(d)  # 表示检测出来几个gtbox
                                # pi[j]对应pred[:,5]中的下标
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if batch_i < 1:
            f = 'test_batch%g_gt.jpg' % batch_i  # filename
            plot_images(imgs, targets, paths=paths, names=names, fname=f)  # ground truth
            f = 'test_batch%g_pred.jpg' % batch_i
            plot_images(imgs, output_to_target(output, width, height), paths=paths, names=names, fname=f)  # predictions

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Print speeds
    if verbose or save_json:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and map and len(jdict):
        print('\nCOCO mAP with pycocotools...')
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
        except:
            print('WARNING: pycocotools must be installed with numpy==1.17 to run correctly. '
                  'See https://github.com/cocodataset/cocoapi/issues/356')

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/fire.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    print(opt)

    # task = 'test', 'study', 'benchmark'
    if opt.task == 'test':  # (default) test normally
        test(opt.cfg,
             opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment)

    elif opt.task == 'benchmark':  # mAPs at 256-640 at conf 0.5 and 0.7
        y = []
        for i in list(range(256, 640, 128)):  # img-size
            for j in [0.6, 0.7]:  # iou-thres
                t = time.time()
                r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, i, opt.conf_thres, j, opt.save_json)[0]
                y.append(r + (time.time() - t,))
        np.savetxt('benchmark.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')
