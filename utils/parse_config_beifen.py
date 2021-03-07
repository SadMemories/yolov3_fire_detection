import os

import numpy as np


def parse_model_cfg(path):
    # Parse the yolo *.cfg file and return module definitions path may be 'cfg/yolov3.cfg', 'yolov3.cfg', or 'yolov3'
    if not path.endswith('.cfg'):  # add .cfg suffix if omitted
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):  # add cfg/ prefix if omitted
        path = 'cfg' + os.sep + path

    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    # rstrip()删除指定字符串末尾的字符(默认为空格) lstrip()用于去除字符串左边的字符(默认为空格) 直接用x.strip()即可
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    mdefs = []  # module definitions
    for line in lines:
        # [ 开头的表示一个模块
        if line.startswith('['):  # This marks the start of a new block
            mdefs.append({})  # 添加一个空字典
            mdefs[-1]['type'] = line[1:-1].rstrip()  # type: xxx
            if mdefs[-1]['type'] == 'convolutional':
                # 预先设置bn为0 原因：预先设置针对的是YOLO层之前的一层 这一层不使用bn 其他的con有bn 并且后序在else会覆盖预先设置的
                mdefs[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if key == 'anchors':  # return nparray
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors
            elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):  # return array
                mdefs[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                # TODO: .isnumeric() actually fails to get the float case
                # isnumeric()检测字符串是否只由数字构成 返回bool类型
                if val.isnumeric():  # return int or float
                    # (int(val)-float(val))==0表示val是整形的
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val  # return string

    # Check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'probability']

    f = []  # fields
    for x in mdefs[1:]:  # net层不算入模块中
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields
    assert not any(u), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (u, path)

    return mdefs


def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options
