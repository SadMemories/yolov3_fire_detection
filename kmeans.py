import numpy as np


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between N boxes and K clusters.
    :param boxes: numpy array of shape (n, 2) where n is the number of box, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (n, k) where k is the number of clusters
    """
    N = box.shape[0]
    K = clusters.shape[0]

    # np.minimun用于比较两个array 在每一个位置返回小的值
    iw = np.minimum(
        np.broadcast_to(box[:, np.newaxis, 0], (N, K)),  # (N, 1) -> (N,K)
        np.broadcast_to(clusters[np.newaxis, :, 0], (N, K))   # (1, K) -> (N,K)
    )
    ih = np.minimum(
        np.broadcast_to(box[:, np.newaxis, 1], (N, K)),
        np.broadcast_to(clusters[np.newaxis, :, 1], (N, K))
    )

    if np.count_nonzero(iw == 0) > 0 or np.count_nonzero(ih == 0) > 0:
        raise ValueError("some box has no area")

    intersection = iw * ih  # (N, K)
    boxes_area = np.broadcast_to((box[:, np.newaxis, 0] * box[:, np.newaxis, 1]), (N, K))
    clusters_area = np.broadcast_to((clusters[np.newaxis, :, 0] * clusters[np.newaxis, :, 1]), (N, K))

    iou_ = intersection / (boxes_area + clusters_area - intersection + 1e-7)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean(np.max(iou(boxes, clusters), axis=1))


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))  # 构建一个距离矩阵
    last_cluster = np.zeros((rows,))

    np.random.seed()

    # np.random.choice(rows, k, replace=False)表示从rows中取k个样本 replace为false表示不放回取样 为True表示放回取样
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    iter_num = 0
    while True:
        iter_num += 1
        print("Iteration: %d" % iter_num)

        distances = 1 - iou(boxes, clusters)  # 计算距离矩阵
        nearest_cluster = np.argmin(distances, axis=1)  # argmin返回的是最小值所在的下标

        if (last_cluster == nearest_cluster).all():
            break

        for cluster in range(k):
            if len(boxes[nearest_cluster == cluster]) == 0:
                print("Cluster %d is zero size" % cluster)

                clusters[cluster] = boxes[np.random.choice(rows, 1, replace=False)]
                continue
            clusters[cluster] = dist(boxes[nearest_cluster == cluster], axis=0)

        last_cluster = nearest_cluster

    return clusters


