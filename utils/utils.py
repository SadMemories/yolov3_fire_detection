import glob
import math
import os
import random
import shutil
import subprocess
import time
from copy import copy
from pathlib import Path
from sys import platform

import cv2
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from . import torch_utils  # , google_utils

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def init_seeds(seed=0):
    random.seed(seed)  # 指定随机种子
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


# 这一段暂时不知道什么意思
def check_git_status():
    if platform in ['linux', 'darwin']:
        # Suggest 'git pull' if repo is out of date
        s = subprocess.check_output('if [ -d .git ]; then git fetch && git status -uno; fi', shell=True).decode('utf-8')
        if 'Your branch is behind' in s:
            print(s[s.find('Your branch is behind'):s.find('\n\n')] + '\n')


def check_file(file):
    # Searches for file if not found locally
    if os.path.isfile(file):
        return file
    else:
        # glob.glob()递归寻找file文件
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        return files[0]  # return first file if multiple found


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    # bincount()解释：https://blog.csdn.net/xlinsist/article/details/51346523
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    # Prepend gridpoint count (for uCE trianing)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    # clamp_将小于0的设置为0 将大于img_shape[1]的设置为1
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    # 从大到小conf排序 argsort返回的下标
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls) # 0

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    # 在这里tp.shape[1]=1表示只对应一个number iou thresholds
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        # 这里n_gt和n_p始终是相等的 因为火焰种类只有一种
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)  # 返回axis=0轴的给定和 表示预测错误的数量
            tpc = tp[i].cumsum(0)  # 返回axis=0轴的给定和 表示预测正确的数量

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # ax.plot(recall, precision)
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1.01)
            # ax.set_ylim(0, 1.01)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    # box1是(4,m) 但是box2是(m,4) 所以下面需要使用.t()来对tensor进行转置
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    # clamp(0)是将小于0的值局限在0
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        # cw表示最小闭包区域的宽度 ch表示最小闭包区域的高度
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        # box[2]表示的是x2坐标
        return (box[2] - box[0]) * (box[3] - box[1])

    # .t()表示对tensor进行转置
    # box1.t()变为 [4, n] box2.t变为[4, m]
    area1 = box_area(box1.t())  # area1的形状为[1, n]
    area2 = box_area(box2.t())  # area2的形状为[1, m]

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


# wh_iou用于计算anchor和预测框之间的iou 这个iou计算并不依赖与空间位置 此iou只是用于判断anchor对于预测框的大小是否合适
def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    # 第一个冒号表示的是第1维 第一维后面加了一个None 说明在第二维上进行扩展 所以变为[N, 1, 2]
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]  表示在第1维上进行扩展 变为[1, M, 2]
    # prod(dim) 返回输入张量给定维度上每行的积
    # torch.min(wh1, wh2)返回的shape是(N,M,2) prod(2)表示第2维上所有元素的乘积
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    # 返回的shape是(N,M)
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


# p的形状为[3, bs, anchors, gridy, gridx, classes + xywh]
def compute_loss(p, targets, model):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    # 将 lcls lbox lobj 定义为FloatTensor格式
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    # tcls:(3,m) tbox:(3,m,4) indices:(3,4,m) anchors:(3,m,2)
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    # print(np.array(tcls).shape)
    # print(np.array(tbox).shape)
    # print(np.array(indices).shape)
    # print(np.array(anchors).shape)
    # print(np.array(p).shape)
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction=red)  # cls_pw = 1.0
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]), reduction=red)  # obj_pw = 1.0

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    # 这是最新的版本中提供了label smoothing的功能，只能用在多类问题(代码细节没看)
    cp, cn = smooth_BCE(eps=0.0)

    # focal loss(暂时没看)
    g = h['fl_gamma']  # focal loss gamma 0.0
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # per output
    nt = 0  # targets
    # pi的形状为:(3, bs, anchors, grid, grid, classes + xywh) 3表示的是3个yolo_layer
    # bs就是batch中对应的图片的下标
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx gj,gi是方格的左上角坐标
        # print(gj.shape)  # torch.size([14])
        # print(gj)  # 示例：tensor([ 6,  3, 12, 14,  7, 10,  9,  1,  3, 12, 14,  7,  3, 12], device='cuda:0')
        # print(b.shape)  # torch.size([14])
        # pi[..., 0]表示舍弃了最后一维 所以tobj为[bs,anchors,grid,grid]
        tobj = torch.zeros_like(pi[..., 0])  # target obj

        nb = b.shape[0]  # number of targets
        if nb:
            nt += nb  # cumulative(累计) targets
            # 找到对应目标的classes+xywh
            # 寻找b a gj gi中相对应的网格中的预测值
            # ps: (nb, 6)
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # GIoU
            pxy = ps[:, :2].sigmoid()  # xcenter ycenter
            pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i]  # w,h
            pbox = torch.cat((pxy, pwh), 1)  # predicted box (nb,4)
            # .t()表示对tensor进行转置
            # giou是(1, m) 这里使用的是GIOU
            # 原本是GIOU=True 现在改成了DIOU=True
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, DIoU=True)  # giou(prediction, target)
            # print(giou.shape)
            lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 - giou).mean()  # giou loss

            # Obj
            # model.gr占一部分
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Class
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn)  # targets
                t[range(nb), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

    # h['giou'] h['obj'] h['cls']都是作者指定的 但是不知道作者是如何得出来的
    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    if red == 'sum':
        bs = tobj.shape[0]  # batch size
        g = 3.0  # loss gain
        lobj *= g / bs
        if nt:
            lcls *= g / nt / model.nc
            lbox *= g / nt

    loss = lbox + lobj + lcls
    # print(lbox)
    # print(lobj)
    # print(lcls)
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    nt = targets.shape[0]  # 多少个 labels 假设一共25个labels
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets

    style = None
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    # j表示的是yolo_layer在module_list的第几层
    for i, j in enumerate(model.yolo_layers):
        # anchor_vec是一个[3, 2] 3表示anchor的数量 2表示 wh anchor中的wh是相对于网格的
        anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        '''
        这里使用[3, 2, 3, 2]的原因：首先需要知道p的形式为[3, bs, anchors, gridy, gridx, classes + xywh](gridy gridx的顺序的原因可以看models.py文件中第240行)
        则p[i]的形式为[bs, anchors, gridy, gridx, classes + xywh] 因为这里面显示gridy，后面才是gridx，gridy对应
        的是height，gridx对应width，所以需要将gridx和gridy进行调换
        '''
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain (1,1,13,13,13,13)
        # print(p[i].shape)  [bs, 3, xx, xx, 6]
        # print(gain[2:])  # [xx,xx,xx,xx]
        na = anchors.shape[0]  # number of anchors 3
        # at = [[0,...,0],[1,...,1],[2,...,2]] (3, 25)(25是假设的labels)
        at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)

        # Match targets to anchors
        # gain = (1, 1, 13, 13, 13, 13) 后面的四位表示grid的xy的个数
        # targets * gain 表示将targets中的 x,y,w,h 去归一化，即恢复为相对于网格的
        # t大小为(25, 6)(25为假设的labels的个数)
        a, t, offsets = [], targets * gain, 0
        if nt:  # nt表示labels的个数 在这里假设为25
            # r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
            # j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            '''
            t[:, 4:6]表示的是每个label中的w和h
            wh_iou是求出每个anchor与box label相交的iou，如果相交的iou小于等于model.hyp['iou_t'] 说明这个尺度的anchor不适合
            这个gtbox，只有iou大于model.hyp['iou_t']才会保留
            详细解释：https://blog.csdn.net/wqwqqwqw1231/article/details/90667046
            '''
            j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2)) n为剩下的图片数量
            # at[j]表示保留下剩下的图片对应的anchor的下标
            # 原本的t的shape是(25,6) 这里的na=3 经过repeat(3,1,1)之后t变为(3,25,6)了
            # 经过处理之后a变为1维的 t变为二维的(m, 6) m与a是一一对应的
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter
            # print(t.shape)
            # print(a.shape)

            # overlaps
            gxy = t[:, 2:4]  # grid xy 这里的xy是去归一化的 (m, 2) 坐标是相对网格的
            z = torch.zeros_like(gxy)  # z也是(m,2)
            if style == 'rect2':  # style = None 暂时未知含义
                g = 0.2  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                a, t = torch.cat((a, a[j], a[k]), 0), torch.cat((t, t[j], t[k]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1]), 0) * g

            elif style == 'rect4':
                g = 0.5  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

        # Define
        b, c = t[:, :2].long().T  # image, class .T表示转置 b和c都是1*n b表示image c表示class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()  # .long是返回长整数 在这里offsets是等于0的 gij表示位于第几个grid中
        gi, gj = gij.T  # grid xy indices

        # Append
        # .clamp_(min, max)将输入input张量每个元素的夹紧到区间 [min,max]
        # gain[3]对应height维的grid的个数 gain[2]对应width维的grid的个数
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        # print(indices)
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box (m, 4)
        anch.append(anchors[a])  # anchors a中只存储着 0 1 2 anch中存放的是对应anchor的w和h
        tcls.append(c)  # class
        if c.shape[0]:  # if any targets
            assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           model.nc, model.nc - 1, c.max())
    '''
    tcls:(3, m) tbox:(3, m, 4) indices:(3, 4, m) anch:(3, m, 2)
    上面每一个变量的第0维都是3 表示的是3个yolo_layer层
    tcls用于表示种类 tbox对应的 xcenter ycenter w h, xcenter和ycenter是在(0,1)之间 是相对某一个网格左上角的偏移量
    w h是相对网格的 范围不是在(0,1)之间
    indices中最后一维分别表示：label对应的哪一张图片 label对应的哪一个anchor 对应左上角网格x坐标 对应左上角网格y坐标
    anch最后一维表示：w和h，都是相对于网格的
    '''
    # print(np.array(indices).shape)
    return tcls, tbox, indices, anch


# prediction=torch.size([4,10080,6])
def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, multi_label=True, classes=None, agnostic=False):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Settings
    merge = True  # merge for best mAP
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    # prediction[0]表示去除了第0维 剩下就是[10080, 6] 然后shape[1]指的是6 -5是种类
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * prediction.shape[0]  # shape[0]表示每一个batch中图片的数量
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[:, 4]表示的是confidence : 表示的是一张图片中的每一个anchor
        x = x[x[:, 4] > conf_thres]  # confidence 置信度大于conf_thres的anchor留下来
        # 使用min_wh去除anchor过小的框 使用max_wh去除anchor过大的框
        # .all用于判断给定维度内的所有元素是否都为true 如果是 返回True 否则 返回False
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height

        # If none remain process next image
        # 经过上述判断之后 如果没有剩余的anchor 直接跳转到下一张图片
        if not x.shape[0]:
            continue

        # Compute conf
        # 5:表示所有的种类
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            # nonzero()返回元素中非零0的下标 .t()表示转置
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only
            # conf表示的是最大的值 j表示最大的值对应的下标
            # x[:, 5:].max(1) 找出每一个anchor应该预测哪一个种类 返回预测该种类的conf和下标
            conf, j = x[:, 5:].max(1)
            '''
            将原本的x进行替换 经过torch.cat()之后 现在的x的维度变为[num, 6] 6中前4表示x1y1x2y2 
            第5个数表示预测的种类的最大的置信度 第6个数表示预测的哪一个种类
            [conf > conf_thres]表示只留下置信度大于conf_thres的anchor
            '''
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]

        # Filter by class
        if classes:  # none
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        # x.shape[0]表示经过上述步骤之后剩余的anchor的数量
        n = x.shape[0]  # number of boxes
        # 经过筛选之后没有符合条件的box
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        # agnostic为false
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # c表示classes的下标 从0开始 c只是一个n*1为的数组
        # x[:,:4].clone()+c.view(-1,1)*max_wh 表示x中每一维的数都要加上c.view(-1,1)
        # 目前加上c.view(-1,1)*max_wh的含义暂时未知 不过对于单个种类来说加和不加没有影响 因为单个种类下标为0
        # scores的形状为(1,n)
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        # 经过MS过滤后的bouding boxes索引（降序排列）这个索引应该指的是boxes中的索引
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix 形状为[i,n] i表示经过nms处理之后的anchor数量
                # scores[None]的作用是维度加1 比如原本的scores为[1,2,3] 经过scores[None]之后变为[[1,2,3]] 目的是和iou能够进行点乘
                weights = iou * scores[None]  # box weights 形状为[i, n]
                # 下一句作用未知
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                # i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def get_yolo_layers(model):
    bool_vec = [x['type'] == 'yolo' for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]  # [82, 94, 106] for yolov3


def print_model_biases(model):
    # prints the bias neurons preceding each yolo layer
    print('\nModel Bias Summary: %8s%18s%18s%18s' % ('layer', 'regression', 'objectness', 'classification'))
    try:
        multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
        for l in model.yolo_layers:  # print pretrained biases
            if multi_gpu:
                na = model.module.module_list[l].na  # number of anchors
                b = model.module.module_list[l - 1][0].bias.view(na, -1)  # bias 3x85
            else:
                na = model.module_list[l].na
                b = model.module_list[l - 1][0].bias.view(na, -1)  # bias 3x85
            print(' ' * 20 + '%8g %18s%18s%18s' % (l, '%5.2f+/-%-5.2f' % (b[:, :4].mean(), b[:, :4].std()),
                                                   '%5.2f+/-%-5.2f' % (b[:, 4].mean(), b[:, 4].std()),
                                                   '%5.2f+/-%-5.2f' % (b[:, 5:].mean(), b[:, 5:].std())))
    except:
        pass


def strip_optimizer(f='weights/best.pt'):  # from utils.utils import *; strip_optimizer()
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    print('Optimizer stripped from %s' % f)
    torch.save(x, f)


def create_backbone(f='weights/best.pt'):  # from utils.utils import *; create_backbone()
    # create a backbone from a *.pt file
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1
    for p in x['model'].parameters():
        p.requires_grad = True
    s = 'weights/backbone.pt'
    print('%s saved as %s' % (f, s))
    torch.save(x, s)


def coco_class_count(path='../coco/labels/train2014/'):
    # Histogram of occurrences per class
    nc = 80  # number classes
    x = np.zeros(nc, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nc)
        print(i, len(files))


def coco_only_people(path='../coco/labels/train2017/'):  # from utils.utils import *; coco_only_people()
    # Find images with only people
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        if all(labels[:, 0] == 0):
            print(labels.shape[0], file)


def crop_images_random(path='../images/', scale=0.50):  # from utils.utils import *; crop_images_random()
    # crops images into random squares up to scale fraction
    # WARNING: overwrites images!
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        img = cv2.imread(file)  # BGR
        if img is not None:
            h, w = img.shape[:2]

            # create random mask
            a = 30  # minimum size (pixels)
            mask_h = random.randint(a, int(max(a, h * scale)))  # mask height
            mask_w = mask_h  # mask width

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            cv2.imwrite(file, img[ymin:ymax, xmin:xmax])


def coco_single_class_labels(path='../coco/labels/train2014/', label_class=43):
    # Makes single-class coco datasets. from utils.utils import *; coco_single_class_labels()
    if os.path.exists('new/'):
        shutil.rmtree('new/')  # delete output folder
    os.makedirs('new/')  # make new output folder
    os.makedirs('new/labels/')
    os.makedirs('new/images/')
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        with open(file, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
        i = labels[:, 0] == label_class
        if any(i):
            img_file = file.replace('labels', 'images').replace('txt', 'jpg')
            labels[:, 0] = 0  # reset class to 0
            with open('new/images.txt', 'a') as f:  # add image to dataset list
                f.write(img_file + '\n')
            with open('new/labels/' + Path(file).name, 'a') as f:  # write label
                for l in labels[i]:
                    f.write('%g %.6f %.6f %.6f %.6f\n' % tuple(l))
            shutil.copyfile(src=img_file, dst='new/images/' + Path(file).name.replace('txt', 'jpg'))  # copy images


def kmean_anchors(path='./data/coco64.txt', n=9, img_size=(640, 640), thr=0.20, gen=1000):
    # Creates kmeans anchors for use in *.cfg files: from utils.utils import *; _ = kmean_anchors()
    # n: number of anchors
    # img_size: (min, max) image size used for multi-scale training (can be same values)
    # thr: IoU threshold hyperparameter used for training (0.0 - 1.0)
    # gen: generations to evolve anchors using genetic algorithm
    from utils.datasets import LoadImagesAndLabels

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        iou = wh_iou(wh, torch.Tensor(k))
        max_iou = iou.max(1)[0]
        bpr, aat = (max_iou > thr).float().mean(), (iou > thr).float().mean() * n  # best possible recall, anch > thr
        print('%.2f iou_thr: %.3f best possible recall, %.2f anchors > thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, IoU_all=%.3f/%.3f-mean/best, IoU>thr=%.3f-mean: ' %
              (n, img_size, iou.mean(), max_iou.mean(), iou[iou > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    def fitness(k):  # mutation fitness
        iou = wh_iou(wh, torch.Tensor(k))  # iou
        max_iou = iou.max(1)[0]
        return (max_iou * (max_iou > thr).float()).mean()  # product

    # Get label wh
    wh = []
    dataset = LoadImagesAndLabels(path, augment=True, rect=True)
    nr = 1 if img_size[0] == img_size[1] else 10  # number augmentation repetitions
    for s, l in zip(dataset.shapes, dataset.labels):
        wh.append(l[:, 3:5] * (s / s.max()))  # image normalized to letterbox normalized wh
    wh = np.concatenate(wh, 0).repeat(nr, axis=0)  # augment 10x
    wh *= np.random.uniform(img_size[0], img_size[1], size=(wh.shape[0], 1))  # normalized to pixels (multi-scale)
    wh = wh[(wh > 2.0).all(1)]  # remove below threshold boxes (< 2 pixels wh)

    # Kmeans calculation
    from scipy.cluster.vq import kmeans
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.Tensor(wh)
    k = print_results(k)

    # # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.tight_layout()
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    for _ in tqdm(range(gen), desc='Evolving anchors'):
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            print_results(k)
    k = print_results(k)

    return k


def print_mutation(hyp, results, bucket=''):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        os.system('gsutil cp gs://%s/evolve.txt .' % bucket)  # download evolve.txt

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    np.savetxt('evolve.txt', x[np.argsort(-fitness(x))], '%10.3g')  # save sort by fitness

    if bucket:
        os.system('gsutil cp evolve.txt gs://%s' % bucket)  # upload evolve.txt


def apply_classifier(x, model, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    w = [0.0, 0.01, 0.99, 0.00]  # weights for [P, R, mAP, F1]@0.5 or [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def output_to_target(output, width, height):
    """
    Convert a YOLO model output to target format
    [batch_id, class_id, x, y, w, h, conf]
    """
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    targets = []
    for i, o in enumerate(output):
        if o is not None:
            for pred in o:
                box = pred[:4]
                w = (box[2] - box[0]) / width
                h = (box[3] - box[1]) / height
                x = box[0] / width + w / 2
                y = box[1] / height + h / 2
                conf = pred[4]
                cls = int(pred[5])

                targets.append([i, cls, x, y, w, h, conf])

    return np.array(targets)


# Plotting functions ---------------------------------------------------------------------------------------------------
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_wh_methods():  # from utils.utils import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(x, ya, '.-', label='yolo method')
    plt.plot(x, yb ** 2, '.-', label='^2 power method')
    plt.plot(x, yb ** 2.5, '.-', label='^2.5 power method')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.legend()
    fig.tight_layout()
    fig.savefig('comparison.png', dpi=200)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    # print(fname)
    if os.path.isfile(fname):  # do not overwrite
        return None

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    # 去归一化
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _(channel), height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            gt = image_targets.shape[1] == 6  # ground truth if no conf column
            conf = None if gt else image_targets[:, 6]  # check for confidence presence (gt vs pred)

            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:  # 0.3 conf thresh
                    label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname is not None:
        mosaic = cv2.resize(mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic


def plot_lr_scheduler(optimizer, scheduler, epochs=300):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.tight_layout()
    plt.savefig('LR.png', dpi=200)


def plot_test_txt():  # from utils.utils import *; plot_test()
    # Plot test.txt histograms
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # from utils.utils import *; plot_targets_txt()
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


def plot_labels(labels):
    # plot dataset labels
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classees, boxes

    def hist2d(x, y, n=100):
        xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
        hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
        xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
        yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
        return hist[xidx, yidx]

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    ax[0].hist(c, bins=int(c.max() + 1))
    ax[0].set_xlabel('classes')
    ax[1].scatter(b[0], b[1], c=hist2d(b[0], b[1], 90), cmap='jet')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[2].scatter(b[2], b[3], c=hist2d(b[2], b[3], 90), cmap='jet')
    ax[2].set_xlabel('width')
    ax[2].set_ylabel('height')
    plt.savefig('labels.png', dpi=200)


def plot_evolution_results(hyp):  # from utils.utils import *; plot_evolution_results(hyp)
    # Plot hyperparameter evolution results in evolve.txt
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    # weights = (f - f.min()) ** 2  # for weighted results
    fig = plt.figure(figsize=(12, 10), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(4, 5, i + 1)
        plt.plot(mu, f.max(), 'o', markersize=10)
        plt.plot(y, f, '.')
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)


def plot_results_overlay(start=0, stop=0):  # from utils.utils import *; plot_results_overlay()
    # Plot training results files 'results*.txt', overlaying train and val losses
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5', 'val', 'val', 'val', 'Recall', 'F1']  # legends
    t = ['GIoU', 'Objectness', 'Classification', 'P-R', 'mAP-F1']  # titles
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                if i in [0, 1, 2]:
                    y[y == 0] = np.nan  # dont show zero loss values
                ax[i].plot(x, y, marker='.', label=s[j])
            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None  # add filename
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


def plot_results(start=0, stop=0, bucket='', id=()):  # from utils.utils import *; plot_results()
    # Plot training 'results*.txt' as seen in https://github.com/ultralytics/yolov3#training
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val GIoU', 'val Objectness', 'val Classification', 'mAP@0.5', 'F1']
    if bucket:
        os.system('rm -rf storage.googleapis.com')
        files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
    else:
        files = glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')
    for f in sorted(files):
        try:
            results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
            n = results.shape[1]  # number of rows
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # dont show zero loss values
                    # y /= y[0]  # normalize
                ax[i].plot(x, y, marker='.', label=Path(f).stem, linewidth=2, markersize=8)
                ax[i].set_title(s[i])
                # if i in [5, 6, 7]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except:
            print('Warning: Plotting error for %s, skipping file' % f)

    ax[1].legend()
    fig.savefig('results.png', dpi=200)
