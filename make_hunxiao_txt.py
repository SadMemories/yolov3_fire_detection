'''
脚本功能：
1. 将易混淆的火焰图片加入到train.txt训练集中
'''

import os
import random

root_path = 'data/hunxiao'
total_img = os.listdir(root_path)

num = len(total_img)

ftrain = open('data/ImageSets/train.txt', 'a')

for i in range(num):
    name = total_img[i][:-4] + '\n'
    ftrain.write(name)

ftrain.close()