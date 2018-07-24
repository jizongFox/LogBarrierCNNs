#coding=utf8
import numpy as np
import maxflow
from scipy.misc import imread
from matplotlib import pyplot as plt
import networkx as nx
import cv2
import math

lumda = 1
k = 1


def create_graph():
    img = imread("441170.jpeg")
    row, column = img.shape
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((img.shape))

    # 　　　　　　　　  # 画素値の差をノード間の重みにするのでパディング
    pad_im = np.pad(img, ((0, 0), (1, 1)), 'constant', constant_values=0)
    weights = np.zeros((img.shape))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            weights[i, j] = lumda * math.exp((-k) * abs(pad_im[i, j] - pad_im[i, j + 1]))
    structure = np.zeros((3, 3))
    structure[1, 2] = 1
    g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=True)

    pad_im = np.pad(img, ((1, 1), (0, 0)), 'constant', constant_values=0)
    weights = np.zeros((img.shape))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            weights[i, j] = lumda * math.exp((-k) * abs(pad_im[i, j] - pad_im[i + 1, j]))
    structure = np.zeros((3, 3))
    structure[2, 1] = 1
    g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=True)
    # 　　　　　　　　  # マスク画像をロード
    gro = imread('%05d.png' % 200)
    x, y = np.where(gro > 10)
    for i in range(x.shape[0]):
        if img[x[i], y[i]] >= 100:
            temp = x[i] * column + y[i]
            g.add_tedge(temp, 1000000000000, 0)
            label[x[i], y[i]] = 1
    x, y = np.where(img == 0)
    for i in range(x.shape[0]):
        temp = x[i] * column + y[i]
        g.add_tedge(temp, 0, 100000)
    return nodeids, g

if __name__ == '__main__':
    nodeids, g = create_graph()
    g.maxflow()
    sgm = g.get_grid_segments(nodeids)
    img = np.int_(np.logical_not(sgm))
    m = np.zeros((1040, 1392, 3))

    m[:, :, 0] = np.int_(np.logical_not(sgm))
    m = m.astype('uint8') * 255
    img = cv2.imread("result200.tif")
    dst = cv2.addWeighted(img, 0.5, m, 0.5, 0)
    plt.imshow(dst), plt.show()