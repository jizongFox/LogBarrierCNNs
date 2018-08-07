
# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F

import numpy as np
#
# img = np.array([[0, 1, 2]]).T + np.array([0, 4, 10])
# pad_im = np.pad(img, ((0, 0), (1, 1)), 'constant', constant_values=0)
# weights = np.zeros((img.shape))
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         weights[i, j] = np.abs(pad_im[i, j] - pad_im[i, j + 1])
#         # weights[i, j] = np.exp(-1 * np.abs(pad_im[i, j] - pad_im[i, j + 1]))
# structure = np.zeros((3, 3))
# structure[1, 2] = 1
# print(img, '\n')
#
# print(structure, '\n')
#
# print(weights, '\n')


img = np.array([[0, 1, 2]]).T + np.array([0, 2, 4])

weights = np.zeros((img.shape))
weights[:, :-1] = np.abs(img[:, :-1] - img[:, 1:])

structure = np.zeros((3, 3))
structure[1, 2] = 1

print('\n--- Left-right ---\n')
print('\nImage:\n', img)
print('\nStructure:\n', structure)
print('\nWeights:\n', weights)

# add_grid_edge(...)

weights = np.zeros((img.shape))
weights[:-1, :] = np.abs(img[:-1, :] - img[1:, :])

structure = np.zeros((3, 3))
structure[2, 1] = 1

print('\n--- Up-down ---\n')
print('\nImage:\n', img)
print('\nStructure:\n', structure)
print('\nWeights:\n', weights)

# add_grid_edge(...)