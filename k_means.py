import glob
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = '/home/wwj/FireDetection/yolov3-archive-v2/data/Annotations'  # xml所在的路径
CLUSTERS = 9  # 聚类的anchor的数量
BBOX_NORMALIZE = False  # 训练时是否使用了归一化


def show_cluster(data, cluster, max_points=2000):
    if len(data) > max_points:
        idx = np.random.choice(len(data), max_points)
        data = data[idx]
    plt.scatter(data[:, 0], data[:, 1], s=5, c='green')
    plt.scatter(cluster[:, 0], cluster[:, 1], c='red', s=100, marker="^")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Bounding and anchor distribution")
    plt.savefig("cluster.png")
    plt.show()


def show_width_height(data, cluster, bins=50):

    if data.dtype != np.float32:
        data = data.astype(np.float32)
    width = data[:, 0]
    height = data[:, 1]
    ratio = height / width

    plt.figure(1, figsize=(20, 6))
    plt.subplot(131)
    plt.hist(width, bins=bins, color='green')
    plt.xlabel('width')
    plt.ylabel('number')
    plt.title('Distribution of Width')

    plt.subplot(132)
    plt.hist(height, bins=bins, color='blue')
    plt.xlabel('Height')
    plt.ylabel('Number')
    plt.title('Distribution of Height')

    plt.subplot(133)
    plt.hist(ratio, bins=bins, color='magenta')
    plt.xlabel('Height / Width')
    plt.ylabel('number')
    plt.title('Distribution of aspect ratio(Height / Width)')
    plt.savefig("shape-distribution.png")
    plt.show()


def load_dataset(path, normalize=True):
    dataset = []

    for xml_file in glob.glob('{}/*.xml'.format(path)):
        tree = ET.parse(xml_file)

        height = int(tree.findtext('./size/height'))  # 图片高度
        width = int(tree.findtext('./size/width'))  # 图片宽度

        for obj in tree.iter('object'):  # iter用来寻找符合要求的tag
            if normalize:
                xmin = int(obj.findtext('bndbox/xmin')) / float(width)
                xmax = int(obj.findtext('bndbox/xmax')) / float(width)
                ymin = int(obj.findtext('bndbox/ymin')) / float(height)
                ymax = int(obj.findtext('bndbox/ymax')) / float(height)
            else:
                xmin = int(obj.findtext('bndbox/xmin'))
                xmax = int(obj.findtext('bndbox/xmax'))
                ymin = int(obj.findtext('bndbox/ymin'))
                ymax = int(obj.findtext('bndbox/ymax'))
            if (xmax - xmin) == 0 or (ymax - ymin) == 0:
                print(xml_file)  # 打印出这个文件的信息 因为这个文件可能存在错误
                continue
            dataset.append([xmax - xmin, ymax - ymin])  # dataset中对应[width, height]

    return np.array(dataset)


def sort_cluster(cluster):
    if cluster.dtype != np.float32:
        cluster = cluster.astype(np.float32)

    area = cluster[:, 0] * cluster[:, 1]
    cluster = cluster[area.argsort()]
    ratio = cluster[:, 1:2] / cluster[:, 0:1]
    return np.concatenate([cluster, ratio], axis=-1)


print('Start to load xml data from {}'.format(ANNOTATIONS_PATH))
data = load_dataset(ANNOTATIONS_PATH, BBOX_NORMALIZE)

print('Start to do kmeans, please wait a moment')
out = kmeans(data, k=CLUSTERS)

out_sorted = sort_cluster(out)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))

show_cluster(data, out, max_points=2000)

if out.dtype != np.float32:
    out = out.astype(np.float32)

print("Recommanded aspect ratios(width/height)")
print("Width    Height   Height/Width")
for i in range(len(out_sorted)):
    print("%.3f      %.3f     %.1f" % (out_sorted[i, 0], out_sorted[i, 1], out_sorted[i, 2]))
show_width_height(data, out, bins=50)
