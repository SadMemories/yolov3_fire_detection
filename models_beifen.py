from utils.google_utils import *
from utils.layers import *
from utils.parse_config import *

ONNX_EXPORT = False


def create_modules(module_defs, img_size, cfg):
    # Constructs module list of layer blocks from module configuration in module_defs

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size  # expand if necessary
    # pop(0)表示将cfg中的[net]抛出
    _ = module_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels 图片的通道
    module_list = nn.ModuleList()
    # routs统计哪些特征层的输出会被后续的层所使用到(可能是特征融合 也可能是拼接)
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']  # 判断是否使用了BN
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       # mdef['pad']是bool类型 判断是否需要pad
                                                       padding=k // 2 if mdef['pad'] else 0,
                                                       groups=mdef['groups'] if 'groups' in mdef else 1,
                                                       bias=not bn))  # 如果使用BN 那么bias就不起作用 所以使用not bn
            else:  # multiple-size conv
                # MaxConv2d未知含义(到目前没有接触过)
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
                                                          out_ch=filters,
                                                          k=k,
                                                          stride=stride,
                                                          bias=not bn))

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                # 如果该卷积层没有bn 意味着该层为YOLO的predictor
                routs.append(i)  # detection output (goes into yolo layer)
            # 在YOLOv3-spp中 除了三个predictor的activation为linear 其他的都为leaky
            if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())

        elif mdef['type'] == 'BatchNorm2d':  # 在YOLOv3-spp中未使用到
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
            if i == 0 and filters == 3:  # normalize RGB image
                # imagenet mean and var https://pytorch.org/docs/stable/torchvision/models.html#classification
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mdef['type'] == 'maxpool':  # 只有spp模块中使用了maxpool层
            k = mdef['size']  # kernel size
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                # 这里直接使用了modules=maxpool 而没有使用modules.add_module() 是因为maxpool只出现在spp模块
                # 而spp模块中每一个maxpool都独属于一个模块
                modules = maxpool

        elif mdef['type'] == 'upsample':  # 每一个上采样属于一个模块
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))  # img_size = (320, 192)
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer 用于之后concatenate
            layers = mdef['layers']
            '''
            当l大于0的时候使用l+1的原因：因为在第14行已经创建了一个输入为3通道的层了 所以要加1
            filters表示输出的通道的数量
            '''
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            # layers中是存在小于0的数的
            modules = FeatureConcat(layers=layers)

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            # 这里的weight在yolov3-spp中没有起作用
            modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1  # 记录第几个YOLO层 [0, 1, 2]
            '''
            stride是指的YOLO层的输入相对于原图像缩小的比例 在YOLOv3-spp中 第一个YOLO层的输入为16*16 
            相对于原来的512*512 缩小的比例为32倍，同样 第二个YOLO层的输入为32*32 相对于原来缩小了16倍
            第三个YOLO层的输入为64*64 相对于原来缩小了8倍
            '''
            stride = [32, 16, 8]  # P5, P4, P3 strides
            # if语句在YOLOv3-spp中未使用到
            if any(x in cfg for x in ['panet', 'yolov4', 'cd53']):  # stride order reversed
                stride = list(reversed(stride))
            # 在yolov3-spp中未使用到
            layers = mdef['from'] if 'from' in mdef else []
            # mask代表的使用的是第几个anchors
            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
                                nc=mdef['classes'],  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1, 2...
                                layers=layers,  # output layers  在YOLOv3-spp中未用到(未验证)
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            # try对应的YOLOlayer上的上一层 也就是predictor
            try:
                j = layers[yolo_index] if 'from' in mdef else -1  # layers[]在YOLOv3-spp中未用到
                # If previous layer is a dropout layer, get the one before
                if module_list[j].__class__.__name__ == 'Dropout':
                    j -= 1
                # j的值为-1 module_list[-1][0]表示的是卷积层 由于predictor层没有使用BN 所有是有bias项的
                bias_ = module_list[j][0].bias  # shape(255,)
                # modules.no表示YOLO层中预测的维度 modules.na表示yolo层的anchor的个数
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
                bias[:, 4] += -4.5  # obj
                bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        elif mdef['type'] == 'dropout':
            perc = float(mdef['probability'])
            modules = nn.Dropout(p=perc)
        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    """
    此类是对predictor的输出进行一个后处理
    """
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)  # 3 * 2
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices 在YOLOv3-spp中未用到
        self.stride = stride  # layer stride 对应[32,16,8]
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85) 针对每个anchor 会预测多少个参数
        # nx ny对应的是预测特征图的宽度和高度 ng是gird cell的size(待验证)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        # 传入的self.anchors是针对原图像的尺度 所以要除以stride 转变为缩放之后的尺度
        # self.anchor_vec的shape为[3,2] 3代表的anchor的数量 2代表的是wh(在pars_config文件中处理的)
        self.anchor_vec = self.anchors / self.stride
        # 下面对应的5个维度分别是：batch_size na grid_h grid_w wh
        # 值为1的维度对应的不是固定值 会在后序中变化
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device='cpu'):  # 传入的grid的宽度和高度
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        # 构造每个cell处的anchor的xy的偏移量(在feature map上)
        # https://www.bilibili.com/video/BV1t54y1C7ra?t=1032&p=3 视频有讲解
        '''
        训练的时候不需要回归到最终预测的boxes，这句话是说在训练计算损失的时候只是用到了偏移量 比如说tx ty
        没有用到相对于图片的整个坐标
        '''
        if not self.training:  # 训练的时候不需要回归到最终预测boxes
            # torch.meshgrid用于生成网格 两个输出张量的行数为第一个输入张量的元素个数 列数为第二个输入张量的元素个数 用于生成坐标
            # yv,xv即对应着网格的纵坐标和横坐标
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            # self.grid是针对预测而言的
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    # p对应的是predictor预测的参数 即最后一个conv2d层
    def forward(self, p, out):
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.index, self.nl  # index in layers, number of layers
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape  # batch-size, predictor_param(255), grid(13), grid(13)
            # 如果是不等于的话 说明grid size 是发生的变化的
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)  # 重新生成grid参数

            # outputs and weights
            # w = F.softmax(p[:, -n:], 1)  # normalized weights
            w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)
            # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

            # weighted ASFF sum
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * \
                         F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)

        elif ONNX_EXPORT:
            bs = 1  # batch size
        else:
            # p为[bs, channel, height, width] 所以height对应的为ny
            bs, _, ny, nx = p.shape  # bacth_size, predict_param(255), grid(13), grid(13)
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        # view: (bs, 255, 13, 13) --> (bs, 3, 85, 13, 13)
        # permute: (bs, 3, 85, 13, 13) --> (bs, 3, 13, 13, 85)
        # 由于执行了permute之后 数据再内存中不连续了 所以需要使用contiguous将其变为连续的
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            io = p.clone()  # inference output
            '''
            io[..., :2]表示在最后一个维度的前两个值 最后一个维度一共有85个值 前两个值表示x和y
            这里的xy是相对于cell的偏移量 经过Sigmoid处理后，需要加上grid(grid是每个cell左上角的坐标)相对于整副图片的xy坐标
            '''
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            # 论文中 bw = pw(anchor的宽度)*e^tw 下面的等式就是实例化了论文中的等式
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride  # 乘以缩放比例是将feature map中的坐标映射到原图中
            torch.sigmoid_(io[..., 4:])
            # 这里将3 13 13全部相乘的原因：由于采用的多尺度训练 训练出来的grid的数目不是恒等于13的 会变化
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    # YOLOv3 object detection model

    # verbose表示在实例化的时候需要不需要打印模型每个模块的详细信息 False表示不打印
    # 视频讲解中说 img_size 只在 onnx 模型时起作用(onnx不知道是什么)
    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()
        '''
        self.module_defs本身是一个list list中包含许多的字典 
        每一个字典对应一个模块的内容
        '''
        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg)
        self.yolo_layers = get_yolo_layers(self)
        # torch_utils.initialize_weights(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        # 打印模型的信息 如果verbose为True 则打印详细信息
        self.info(verbose) if not ONNX_EXPORT else None  # print model description

    # x是输入的一个一个数据 已经被打包成了batch
    # augment是数据增强 在test的时候可以使用
    def forward(self, x, augment=False, verbose=False):

        if not augment:
            return self.forward_once(x)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                # y[0]是原本的x y[1]是经过flip和scale的 y[2]是经过scale的
                # [0]表示只传入X X=torch.size([1,25200,6])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale xcenter ycenter width height
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr xcenter
            y[2][..., :4] /= s[1]  # scale

            # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
            #     area = yi[..., 2:4].prod(2)[:, :, None]
            #     if i == 1:
            #         yi *= (area < 96. ** 2).float()
            #     elif i == 2:
            #         yi *= (area > 32. ** 2).float()
            #     y[i] = yi

            y = torch.cat(y, 1)  # y=torch.size([1, xx, 6])
            return y, None

    # x是输入的一个一个数据 已经被打包成了batch (batch, 3, height, width)
    def forward_once(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]  # height, width
        # yolo_out用来收集每一个yololayer层的输出
        # out用来收集每一个模块的输出
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ''

        # Augment images (inference and test only)
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            # x.flip(3)表示在第3维上进行翻转 即width维度 作用是左右翻转
            # torch.cat(..., 0)表示在第0维上拼接 结果会是batch的3倍
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                           torch_utils.scale_img(x, s[1]),  # scale
                           ), 0)

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ['WeightedFeatureFusion', 'FeatureConcat']:  # sum, concat
                # if语句只会一个描述信息
                if verbose:  # verbose = false
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == 'YOLOLayer':
                '''
                在遍历完所有的YOLOLayer之后 yolo_out={list:3}(model.trian下)
                在model.eval()下，yolo_out={list:3}其中每一个元素是{tuple:2}的形式 
                每一个tuple中包含两个tensor 分别为[1,1200(变化的),6]和[1,3,20(变化),20(变化),6]
                '''
                yolo_out.append(module(x, out))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)  # 调用的哪一个类的module暂时未知

            # 这里只有self.routs[i]在存放的原因是 只有self.routs[i]才会用到 因为是用到前面的层数
            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        if self.training:  # train 在mode.train()的时候会自动将self.training设置为true
            return yolo_out
        elif ONNX_EXPORT:  # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else:  # inference or test
            # *yolo_out是将任意个参数以元组的形式传入 yolo_out是一个包含多个module(x, out)的列表 调用的是YOLOlayer中的forword函数
            # 返回的x，p对应着此文件中的268行
            x, p = zip(*yolo_out)  # inference output, training output p={tuple:3}
            x = torch.cat(x, 1)  # cat yolo outputs x 是多个列表 这里是对多个列表在第1维度进行拼接 torch.size([1, 25200, 6])
            # 在推理阶段使用数据增强之后 大小会变为原来的batch的3倍
            if augment:  # de-augment results
                # torch.split表示沿第0维将x分为nb个块
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale 根据351行可知宽度高度是根据s[0]进行缩放的
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr 因为数据增强时进行了水平翻转
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)
            # p={tuple:3} 每一个tuple是[bs, 3, grid, grid, 6]
            return x, p  # x=torch.size([1, 25200, 6])

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        # self.children()只包括网络模块的第一代儿子模块 在本模型中 只包括moduleList一个模块 所以后面使用了[0]
        # for循环遍历moduleList中的每一个子模块
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]  # conv层都是在bn的前一层
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        # [i+1:]表示去除了conv和BN层
                        # 在list前加* 表示将后面的列表拆解开
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None  # yolov3-spp reduced from 225 to 152 layers

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


def get_yolo_layers(model):
    return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:  # rb表示以二进制形式读取
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        # count=3表示读取的第三行 存放的版本信息
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        # count=1表示读取的第一行 存放的训练的图片个数
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training
        # 没加count，表示读取的剩余的所有信息(剩余的所有信息是权重) 剩余的信息不包括上面的两行(不知道为什么不会读入上面的两行)
        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    # [:cutoff]去除了最后一层 最后一层应该是YOLOlayer 所以去除
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        target = weights.rsplit('.', 1)[0] + '.weights'
        save_weights(model, path=target, cutoff=-1)
        print("Success: converted '%s' to '%s'" % (weights, target))

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        target = weights.rsplit('.', 1)[0] + '.pt'
        torch.save(chkpt, target)
        print("Success: converted '%s' to '%s'" % (weights, target))

    else:
        print('Error: extension not supported.')


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip().replace("'", '')
    msg = weights + ' missing, try downloading from https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0'

    if len(weights) > 0 and not os.path.isfile(weights):
        d = {'yolov3-spp.weights': '16lYS4bcIdM2HdmyJBVDOvt3Trx6N3W2R',
             'yolov3.weights': '1uTlyDWlnaqXcsKOktP5aH_zRDbfcDp-y',
             'yolov3-tiny.weights': '1CCF-iNIIkYesIDzaPvdwlcf7H9zSsKZQ',
             'yolov3-spp.pt': '1f6Ovy3BSq2wYq4UfvFUpxJFNDFfrIDcR',
             'yolov3.pt': '1SHNFyoe5Ni8DajDNEqgB2oVKBb_NoEad',
             'yolov3-tiny.pt': '10m_3MlpQwRtZetQxtksm9jqHrPTHZ6vo',
             'darknet53.conv.74': '1WUVBid-XuoUBmvzBVUCBl_ELrzqwA8dJ',
             'yolov3-tiny.conv.15': '1Bw0kCpplxUqyRYAJr9RY9SGnOJbo9nEj',
             'yolov3-spp-ultralytics.pt': '1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4'}

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)
        else:  # download from pjreddie.com
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            r = os.system('curl -f ' + url + ' -o ' + weights)

        # Error check
        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.system('rm ' + weights)  # remove partial downloads
            raise Exception(msg)


if __name__ == '__main__':
    model = Darknet('cfg/yolov3-spp.cfg')
    # print(model.module_list)
    # print(list(model.children())[0])