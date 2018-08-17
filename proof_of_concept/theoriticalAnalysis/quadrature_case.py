# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, sys, os
torch.random.manual_seed(10)
f = lambda x, y: (x)**2 + 10*(y)** 2
g = lambda x, y:  0.6*y+x#if x <0.5 else torch.tensor(0.0)

## to minimize the f(x,y) = x^2 + y^2 under constraint of int(x+y)=1
x = torch.randn(1,requires_grad=True)
y = torch.randn(1,requires_grad=True)


learning_rate = 8e-3
rng = 5
plt.ion()
x_ = np.linspace(-rng, rng, 500)
y_ = np.linspace(-rng, rng, 500)
X_, Y_ = np.meshgrid(x_, y_)
Height = f(X_.reshape(-1),Y_.reshape(-1)).reshape(X_.shape)
plt.contourf(X_, Y_, Height, 30, alpha = 0.6, cmap = plt.cm.hot)
# 绘制等高线
C = plt.contour(X_, Y_, Height, 30, colors = 'black', linewidth = 0.1)
# 显示各等高线的数据标签
plt.clabel(C, inline = True, fontsize = 10)

plt.plot(np.linspace(-rng,rng,500),(2-np.linspace(-rng,rng,500))/0.6)
# plt.plot(np.linspace(-2,2,500),(1+2*np.linspace(-2,2,500)))
plt.xlim([-rng,rng])
plt.ylim([-rng,rng])

plt.show()

p = 1
lam1 = 0
lam2 = 0
lams = True

# lagrangian function L:
beta = 0.9
for i in range (1000000):

    for j in range(1):
        L = f(x,y) + lam1 * (2- g (x,y)) #   + p/2*((2- g (x,y)))**2
        # update (x,y) by finding argmin(L)
        L.backward()

        with torch.no_grad():
            x-= x.grad * learning_rate
            y-= y.grad * learning_rate
            x.grad.zero_()
            y.grad.zero_()

    ## update lam
    if p !=0 :
        lam1 = lam1*beta + p*(float(2- g(x,y).data.numpy()) )*0.1
    else:
        lam1 = lam1 * beta + 1* (float(2 - g(x, y).data.numpy()))
    print((x.item(),y.item()),lam1)
    plt.scatter(x.data.numpy(),y.data.numpy(),)
    plt.pause(0.01)







