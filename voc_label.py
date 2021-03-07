import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = ['train', 'test', 'val']

classes = ["fire"]  # 我们只是检测细胞，因此只有一个类别


def convert(size, box):
    dw = 1. / size[0]  # 归一化
    dh = 1. / size[1]  # 归一化
    x = (box[0] + box[1]) / 2.0  # xcenter
    y = (box[2] + box[3]) / 2.0  # ycenter
    w = box[1] - box[0]  # width
    h = box[3] - box[2]  # height
    x = x * dw  # 归一化
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('data/Annotations/%s.xml' % (image_id))
    out_file = open('data/labels/%s.txt' % (image_id), 'w')  # out_file存放文件的xcenter ycenter width height信息
    tree = ET.parse(in_file)
    root = tree.getroot()  # 这里的root指的是xml文件中的Annotations
    size = root.find('size')
    w = int(size.find('width').text)  # 利用text将其转换为文本形式
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text  # False(没有验证)
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)  # 0
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))  # ()代表元组数据类型
        # print(in_file)
        # print((w,h))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()  # getcwd返回当前工作目录
print(wd)
for image_set in sets:
    if not os.path.exists('data/labels/'):
        os.makedirs('data/labels/')
    image_ids = open('data/ImageSets/%s.txt' % (image_set)).read().strip().split()
    list_file = open('data/%s.txt' % (image_set), 'w')  # list_file中存放的图片的路径
    for image_id in image_ids:
        list_file.write('data/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()