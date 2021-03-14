import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.utils import xyxy2xywh, xywh2xyxy

help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


class LoadImages:  # for inference
    def __init__(self, path, img_size=416):
        path = str(Path(path))  # os-agnostic
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (path, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            # cap.read()按帧读取视频，ret,frame是获cap.read()方法的两个返回值。
            # 其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe=0, img_size=416):
        self.img_size = img_size

        if pipe == '0':
            pipe = 0  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa'  # IP traffic camera
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        # https://answers.opencv.org/question/215996/changing-gstreamer-pipeline-to-opencv-in-pythonsolved/
        # pipe = '"rtspsrc location="rtsp://username:password@192.168.1.64/1" latency=10 ! appsink'  # GStreamer

        # https://answers.opencv.org/question/200787/video-acceleration-gstremer-pipeline-in-videocapture/
        # https://stackoverflow.com/questions/54095699/install-gstreamer-support-for-opencv-python-package  # install help
        # pipe = "rtspsrc location=rtsp://root:root@192.168.0.91:554/axis-media/media.amp?videocodec=h264&resolution=3840x2160 protocols=GST_RTSP_LOWER_TRANS_TCP ! rtph264depay ! queue ! vaapih264dec ! videoconvert ! appsink"  # GStreamer

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, 'Camera Error %s' % self.pipe
        img_path = 'webcam.jpg'
        print('webcam %g: ' % self.count, end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=416):
        self.mode = 'images'
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            cap = cv2.VideoCapture(0 if s == '0' else s)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadImagesAndLabels(Dataset):  # for training/testing
    # path存放的是图片路径
    def __init__(self, path, img_size=416, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, pad=0.0):
        try:

            path = str(Path(path))  # os-agnostic 返回字符串形式的path路径
            parent = str(Path(path).parent) + os.sep  # 去除了最后的文件名称的绝对路径
            if os.path.isfile(path):  # file
                with open(path, 'r') as f:
                    f = f.read().splitlines()
                    # 如果是./ 下面一句话说明images和annotation必须要在data文件夹下
                    f = [x.replace('./', parent) if x.startswith('./') else x for x in f]  # local to global path
            elif os.path.isdir(path):  # folder
                f = glob.iglob(path + os.sep + '*.*')
            else:
                raise Exception('%s does not exist' % path)
            # os.sep会根据训练的系统自己选择分隔符
            # os.path.splitext()将文件名和扩展名分开
            self.img_files = [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats]
        except:
            raise Exception('Error loading data from %s. See %s' % (path, help_url))

        n = len(self.img_files)  # 图片的个数
        assert n > 0, 'No images found in %s. See %s' % (path, help_url)
        # np.floor()用于向下取整
        # bi中存储了每一张图片属于第几个batch 下标从0开始的
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images
        self.batch = bi  # batch index of image 每张图片属于的batch的下标
        self.img_size = img_size
        self.augment = augment  # 是否采用数据增强 训练集时采用
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)

        # Define labels
        # 下面一句话说明了图片文件名不要含有images字符串
        # self.label_files中存放yolo标签所在的路径
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]

        # Read image shapes (wh)
        # sp指向的是my_train_data.shapes文件 其中存放的训练集每张图片的宽度和高度 (好像只有在训练时有用)
        sp = path.replace('.txt', '') + '.shapes'  # shapefile path
        try:
            with open(sp, 'r') as f:  # read existing shapefile
                s = [x.split() for x in f.read().splitlines()]
                # 判断现有的shape文件中的行数(图像个数)是否与当前数据集中图像个数相等
                # 如果不相等则认为是不同的数据集，故重新生成shape文件
                assert len(s) == n, 'Shapefile out of sync'
        except:
            # Image.open(f)读入图片，默认为RGB顺序，读出的变量类型为Image类型, size为(width，height)，但是为彩色三通道图像
            # tqdm库会显示处理的进度
            # exif_size()方法返回了图片的width和height属性
            s = [exif_size(Image.open(f)) for f in tqdm(self.img_files, desc='Reading image shapes')]
            # 将所有图片的shape信息保存在.shape文件中
            # fmt是存储格式 fmt=%g返回的是浮点数
            np.savetxt(sp, s, fmt='%g')  # overwrites existing (if any)
        # width height
        self.shapes = np.array(s, dtype=np.float64)

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        # 如果为ture，训练网络时，会使用类似原图像比例的矩形(让最长边为img_size)，而不是img_size x img_size
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio height/width
            # argsort函数返回的是数组值从小到大的索引值
            # 按照长宽比例进行排序
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]  # bi: batch index 找出第i个batch中所有的图片比例
                mini, maxi = ari.min(), ari.max()  # ari.min()存放的是最小的hw比例 max()是最大的hw比例
                if maxi < 1:  # 如果高/宽小于1(w > h)，将w设为img_size
                    shapes[i] = [maxi, 1]
                elif mini > 1:  # 如果高/宽大于1(w < h)，将h设置为img_size
                    shapes[i] = [1, 1 / mini]
            # 经过上述的for循环之后 shapes都满足 [height, width]的格式
            # 计算每个batch输入网络的shape值
            # np.ceil()表示向上取整 下面的语句是将尺寸转换为32的倍数
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32. + pad).astype(np.int) * 32

        # Cache labels
        # n表示图片的总个数 self.imgs是一个一维数据 self.imgs的作用未知
        self.imgs = [None] * n
        # label: [class, x, y, w, h] 二维数据
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * n
        create_datasubset, extract_bounding_boxes, labels_loaded = False, False, False
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        '''
        Path().parent表示去除了最后一个分隔符/之后表示
        比如说：file=/home/wwj/FireDetection/YOLOv3/data/wang.jpg
        经过Path(file).parent之后变为/home/wwj/FireDetection/YOLOv3/data
        npy文件用于存储重建ndarray所需的数据、图形、dtype和其他信息
        '''
        np_labels_path = str(Path(self.label_files[0]).parent) + '.npy'  # saved labels in *.npy file
        # 如果文件存在 对应np.save(np_labels_path, self.labels)将文件保存的
        if os.path.isfile(np_labels_path):
            s = np_labels_path  # print string
            x = np.load(np_labels_path, allow_pickle=True)
            if len(x) == n:
                self.labels = x
                labels_loaded = True
        else:
            s = path.replace('images', 'labels')

        pbar = tqdm(self.label_files)  # labels_files中存放的都是[cls, x, y, w, h]
        for i, file in enumerate(pbar):
            if labels_loaded:
                l = self.labels[i]
                # np.savetxt(file, l, '%g')  # save *.txt from *.npy file
            else:
                try:
                    with open(file, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except:
                    nm += 1  # print('missing labels for image %s' % self.img_files[i])  # file missing
                    continue
            # 如果标注信息不为空的话 说明l代表的图片中存在object
            # l.shape[0]表示一张图片中有多少个标注框
            if l.shape[0]:
                assert l.shape[1] == 5, '> 5 label columns: %s' % file  # 如果shape[1]大于5 说明文件中信息错误
                assert (l >= 0).all(), 'negative labels: %s' % file  # l中的标签数据应该全部都大于0 如果有小于0的 说明信息错误
                # xcenter ycenter width height都应该经过归一化 都应该小于1
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file

                # 检查每一行，看是否有重复信息
                # np.unique用于除去重复的信息 如果小于原本的l.shape[0]说明有重复信息
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                if single_cls:
                    l[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = l
                nf += 1  # file found

                # Create subdataset (a smaller dataset)
                if create_datasubset and ns < 1E4:
                    if ns == 0:
                        create_folder(path='./datasubset')
                        os.makedirs('./datasubset/images')
                    exclude_classes = 43
                    if exclude_classes not in l[:, 0]:
                        ns += 1
                        # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                        with open('./datasubset/images.txt', 'a') as f:
                            f.write(self.img_files[i] + '\n')

                # Extract object detection boxes for a second stage classifier
                # 训练时没有用到
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)  # make new output folder

                        b = x[1:] * [w, h, w, h]  # box
                        b[2:] = b[2:].max()  # rectangle to square
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                        assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'Failure extracting classifier boxes'
            else:
                ne += 1
                # print('empty labels for image %s' % self.img_files[i])  # file empty
                # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove

            pbar.desc = 'Caching labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                s, nf, nm, ne, nd, n)
        assert nf > 0 or n == 20288, 'No labels found in %s. See %s' % (os.path.dirname(file) + os.sep, help_url)
        if not labels_loaded and n > 1000:
            print('Saving labels to %s for faster future loading' % np_labels_path)
            np.save(np_labels_path, self.labels)  # save for next time

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        if cache_images:  # if training
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image
            for file in tqdm(self.img_files, desc='Detecting corrupted images'):
                try:
                    _ = io.imread(file)
                except:
                    print('Corrupted image detected: %s' % file)

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        # 在训练中为false
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        if self.mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)  # index是一个int类型
            shapes = None

        else:
            # Load image
            # h,w是经过调整之后的 其中有一个值等于img_size img是经过插值之后的图像(且是BGR格式) 其中一边等于img_size
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            # shape存放的height 和 width
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            # self.labels[index]表示index对应的图片中所有的gtbox []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                img, labels = random_affine(img, labels,
                                            degrees=hyp['degrees'],
                                            translate=hyp['translate'],
                                            scale=hyp['scale'],
                                            shear=hyp['shear'])

            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            # 这里的xyxy是未归一化的 xywh也是未归一化的
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]
        # [batch, cls, x, y, w, h]
        labels_out = torch.zeros((nL, 6))  # nl
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        # print(img.shape)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):  # 用于将样本整理为批次
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            # l[:, 0] = i会直接影响到label i表示的是第几张图片
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    # 如果没有使用cache_image 则self.imgs为空
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]  # index图片的路径
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw  opencv读取的图片格式为BGR PIL读取的图片格式为RGB
        r = self.img_size / max(h0, w0)  # resize image to img_size  self.img_size=640
        if r != 1:  # always resize down, only resize up if training with augmentation
            '''
            r<1表示要进行图片缩小，在opencv官方文档中有
            To shrink an image, it will generally look best with cv::INTER_AREA interpolation
            INTER_AREA和INTER_LINEAR的含义没有详细看 有时间看
            interp表示的是插值方法
            '''
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            # 将最大的一边调整为self.img_size尺寸 且h 和 w 的比例不发生变化
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    # np.random.uniform(low, high, size)
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []  # 拼接图像的label信息
    s = self.img_size  # 在trian中是640
    # 随机初始化拼接图像的中心坐标
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    # 从dataset中随机寻找三张图像进行拼接
    # 这里的index代表一张图片的下标 [index]+[...]表示拼接
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        # h,w是经过调整之后的 其中有一个值等于img_size img是经过插值之后的图像(且是BGR格式) 其中一边等于img_size
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            # 创建马赛克图像
            # img.shape[2]表示的是通道数 114表示的是fill_value
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            # 计算截取的图像区域信息(以xc,yc为第二张图像的左下角坐标填充到马赛克图像中，丢弃越界的区域)
            # 其实x2b直接等于x2a-x1a即可 因为x2a-x1a绝对小于等于w(自己认为的)
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第三张图像的右上角坐标填充到马赛克图像中，丢弃越界的区域)
            # 这里的max(xc, w)，我认为应该是w 依据后面的img[y1b:y2b, x1b:x2b]可以x2b一定在img区域内 但是xc可能会大于w
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第四张图像的左上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # 将截取的图像区域填充到马赛克图像的相应位置
        # img是BGR格式 是 hw 形式 所以前面是y 后面是 x
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        # 计算pad(图像边界与马赛克边界的距离，越界的情况为负值)
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        # Labels 获取对应拼接图像的labels信息 self.labels[index]表示index对应的图片中所有的gtbox [class, x, y, w, h]
        x = self.labels[index]
        labels = x.copy()  # 深copy 防止修改原本的信息 labels一般来说是一个二维数据
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            # 计算标注数据在马赛克图像中的
            # x[:, 1]、x[:, 3]分别表示的bbox的xcenter和width 之后乘以w是去归一化
            # labels[:, ]含义：原图像相对于马赛克图像的坐标
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw  # x[:, 1]=xcenter x[:, 3]=width
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)  # labels中保存的是 [x1, y1, x2, y2]

    # cv2.imwrite('wang.jpg', img4)
    # Concat/clip labels
    if len(labels4):  # len(labels4)长度为4
        '''
        np.concatenate(labels4, 0)表示将图片在第0维度上进行
        labels4原本是三维的 经过拼接之后变为了二维的
        第二维中的每一个表示一个[class, x1, y1, x2, y2]
        '''
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        # clip起到裁剪作用 当labels4[:, 1:]存在某些值小于0时，这些小于0的值会被设置为0 当存在某些值大于2*s时，大于0的值会被设置为2*s
        # 改变之后的值保存在 out=labels4[:, 1:]中
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    # img4是4张小图片组成的一张大的图片 labels4存放着若干个[class, x1, y1, x2, y2]
    # img4是 hw 格式
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=self.hyp['degrees'],  # 图像旋转(默认为0)
                                  translate=self.hyp['translate'],  # 图像平移(默认为0)
                                  scale=self.hyp['scale'],  # 默认为0
                                  shear=self.hyp['shear'],  # 图像错切(默认为0)
                                  border=-s // 2)  # border to remove(含义未知)

    return img4, labels4


def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    # shape返回的 h 和 w
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    # h_new/h_old w_new/w_old
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # scaleup表示是否进行数据增强
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    # 将其变换为一张普通图片的高度和宽度
    height = img.shape[0] + border * 2  # img是 hw 形式 所以shape[0]为height
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)  # 返回一个3*3的单位矩阵
    a = random.uniform(-degrees, degrees)  # 在 (-degrees, degrees) 中根据高斯分布随机取值
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)  # 在 (1-scale, 1+scale)中根据高斯分布随机取值
    # s = 2 ** random.uniform(-scale, scale)
    # getRotationMatrix2D：获得图像仿射变换矩阵 返回的是一个2*3的矩阵 R[:2]表示前两行 缩放和旋转
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation(图像平移)
    T = np.eye(3)
    # 图像平移对应3*3的仿射矩阵的第1行第3列和第2行第3列 所以是T[0, 2]和 T[1, 2]
    # border的值表示平移多少像素 正负表示向左向右平移
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear(图像错切)
    S = np.eye(3)
    # 图像错切对应3*3的仿射矩阵的第1行第2列和第2行第1列 所以是S[0,1]和s[1,0]
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    # @表示矩阵 向量乘法
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!(为什么是这个顺序的原因暂时不明白)
    # .any()用于判断矩阵中是否全部为false 如果全部为false则返回false 否则返回true
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        # 这里的dsize应该等于(height, width)(自己的理解) 因为img是opencv读入的 符合 hw 形式
        # 这里不影响 因为img是一个正方体的
        # warpAffine是应用了仿射变换的 输出的img大小为(width, height)
        # cv2.warpAffine的第二个参数必须为2*3的仿射矩阵
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        # cv2.imwrite("wei.jpg", img)

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        # targets[:, [1,2,3,4,1,4,3,2]]将原本的(6, 4)转变为(6, 8)
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        '''
        M.T表示求M矩阵的转置 优先级高于乘法@ 下面一步是对坐标进行仿射变换
        这里用M的转置矩阵的原因：原本的仿射变换应该是M*X 这里的X是列向量 但是xy中(x,y)是横向量 相当于X做了转置
        X做了转置，则M也需要做转置 即M*X=X^T*M^T 所以M就进行了转置 
        '''
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        # 由于bounding box的左上角的坐标一定是(xmin, ymin) 右下角坐标一定是(xmax, ymax) 所以直接使用min和max函数即可
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T  # xy.shape = (n, 4)

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        # clip(0,width)表示小于0的数将其设置为0 大于width的数将其设置为width
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]  # bounding box的width
        h = xy[:, 3] - xy[:, 1]  # bounding box的height
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        '''
        i是一个bool型的矩阵 这里使用 w>4 h>4的原因：去除小于4*4(像素点)的标注框
        设置ar<10的原因：因为ar的计算过程中涉及到除法 可能会有无穷大的数 这里是忽略这些无穷大的数 
        '''
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]  # 只留下满足要求的标注框
        targets[:, 1:5] = xy[i]  # 替换targets中原本的x1 y1 x2 y2坐标

    return img, targets


def cutout(image, labels):
    # https://arxiv.org/abs/1708.04552
    # https://github.com/hysts/pytorch_cutout/blob/master/dataloader.py
    # https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def reduce_img_size(path='../data/sm4/images', img_size=1024):  # from utils.datasets import *; reduce_img_size()
    # creates a new ./images_reduced folder with reduced size images of maximum size img_size
    path_new = path + '_reduced'  # reduced images path
    create_folder(path_new)
    for f in tqdm(glob.glob('%s/*.*' % path)):
        try:
            img = cv2.imread(f)
            h, w = img.shape[:2]
            r = img_size / max(h, w)  # size ratio
            if r < 1.0:
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)  # _LINEAR fastest
            fnew = f.replace(path, path_new)  # .replace(Path(f).suffix, '.jpg')
            cv2.imwrite(fnew, img)
        except:
            print('WARNING: image failure %s' % f)


def convert_images2bmp():  # from utils.datasets import *; convert_images2bmp()
    # Save images
    formats = [x.lower() for x in img_formats] + [x.upper() for x in img_formats]
    # for path in ['../coco/images/val2014', '../coco/images/train2014']:
    for path in ['../data/sm4/images', '../data/sm4/background']:
        create_folder(path + 'bmp')
        for ext in formats:  # ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
            for f in tqdm(glob.glob('%s/*%s' % (path, ext)), desc='Converting %s' % ext):
                cv2.imwrite(f.replace(ext.lower(), '.bmp').replace(path, path + 'bmp'), cv2.imread(f))

    # Save labels
    # for path in ['../coco/trainvalno5k.txt', '../coco/5k.txt']:
    for file in ['../data/sm4/out_train.txt', '../data/sm4/out_test.txt']:
        with open(file, 'r') as f:
            lines = f.read()
            # lines = f.read().replace('2014/', '2014bmp/')  # coco
            lines = lines.replace('/images', '/imagesbmp')
            lines = lines.replace('/background', '/backgroundbmp')
        for ext in formats:
            lines = lines.replace(ext, '.bmp')
        with open(file.replace('.txt', 'bmp.txt'), 'w') as f:
            f.write(lines)


def recursive_dataset2bmp(dataset='../data/sm4_bmp'):  # from utils.datasets import *; recursive_dataset2bmp()
    # Converts dataset to bmp (for faster training)
    formats = [x.lower() for x in img_formats] + [x.upper() for x in img_formats]
    for a, b, files in os.walk(dataset):
        for file in tqdm(files, desc=a):
            p = a + '/' + file
            s = Path(file).suffix
            if s == '.txt':  # replace text
                with open(p, 'r') as f:
                    lines = f.read()
                for f in formats:
                    lines = lines.replace(f, '.bmp')
                with open(p, 'w') as f:
                    f.write(lines)
            elif s in formats:  # replace image
                cv2.imwrite(p.replace(s, '.bmp'), cv2.imread(p))
                if s != '.bmp':
                    os.system("rm '%s'" % p)


def imagelist2folder(path='data/coco_64img.txt'):  # from utils.datasets import *; imagelist2folder()
    # Copies all the images in a text file (list of images) into a folder
    create_folder(path[:-4])
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            os.system('cp "%s" %s' % (line, path[:-4]))
            print(line)


def create_folder(path='./new_folder'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
