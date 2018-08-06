# -*- coding: utf-8 -*-
import numpy as np,pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.misc import imread
import maxflow


class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(in_features=input_size,out_features=1024)
        self.linear2 = nn.Linear(in_features=1024,out_features=2048)
        self.linear3 = nn.Linear(in_features=2048,out_features=4096)
        self.linear4 = nn.Linear(in_features=4096,out_features=input_size)

    def forward(self, input):
        original_size = input.size()
        assert list(input.view(-1).size())[0] ==self.input_size,'size error'
        input = input.view(-1).unsqueeze(0)
        output = F.relu(self.linear1(input))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = F.sigmoid(self.linear4(output))

        return output.reshape(original_size)

def train_pretrain_network(input_image, target_image):
    net = Net(input_image.size)
    input_image = torch.tensor(input_image).float().unsqueeze(0)
    output_image = torch.tensor(target_image).float().unsqueeze(0)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(params=net.parameters(),lr=1e-3)
    for i in range(100):
        optimiser.zero_grad()
        output = net(input_image)
        loss = criterion(output, output_image)
        loss.backward()
        print(loss.item())
        optimiser.step()
    return net

class ADMM(object):
    '''
    to minimize the difference between the size control one and the output of the neural network.

    lagrangian L = v (f(x) - s) + p/2 * (f(x) - s )^2

    su that  the sum of s should be constrained between the Smin and Smax.

    the scaled form of the lagrangian L : p/2 * |f(x)-s + v|^2

    to update f(x) one can use the conventional gradient descent.

    to update s, s+ = argmin p/2 * ( (f(x)+v)^2 + s - 2* (f(x)+v)*s )
                    = argmin  s(1/2 - (f(x)+v))
                    let  a = 1/2 - (f(x)+v)
                s+ = argmin sum ai*si
                st Smin <<sum si << Smax



    '''
    def __init__(self,neural_network, input_image, lower_band,upper_band) -> None:
        super().__init__()
        self.reset()
        self.neural_net = neural_network
        self.lowerband= lower_band
        self.upperband = upper_band
        self.learning_rate = 1e-4
        self.opitimiser = torch.optim.Adam(self.neural_net.parameters(),lr = self.learning_rate)
        self.input_image = input_image
        self.forward_image()
        self.p = 1
        self.neighor=0.2

    def reset(self):
        self.input_image=None
        self.proba=None
        self.p =1
        self.gamma = np.zeros((64,64))
        self.s = np.zeros((64,64))
        self.u = np.zeros((1,64,64))
        self.v = np.zeros((1,64,64))


    def forward_image(self):
        self.proba = self.neural_net(torch.tensor(self.input_image).float().unsqueeze(0))


    def heatmap2segmentation(self, heatmap):
        return heatmap.max(1)[1]

    def update_theta(self):
        for i in range(1):
            self.opitimiser.zero_grad()
            # loss =self.p*(torch.tensor(self.gamma +self.u).float() - self.proba).norm(2)**2
            # above loss is for the gamma
            loss =self.p/2* (self.proba - torch.tensor(self.s - self.v).float()).norm(2)**2
            loss.backward()
            print(loss.item())
            self.opitimiser.step()
            self.forward_image()


    def update_gamma(self):
        unary_term_gamma_1 = np.multiply(
            (0.5 - self.proba.data.numpy() + self.u), 1)
        unary_term_gamma_0 = np.zeros(unary_term_gamma_1.shape)
        neighbor_term = self.neighor
        g = maxflow.Graph[float](0, 0)

        # Add the nodes.
        nodeids = g.add_grid_nodes(list(self.gamma.shape))
        g.add_grid_edges(nodeids, neighbor_term)
        g.add_grid_tedges(nodeids, (unary_term_gamma_0[0]).squeeze(),
                          (unary_term_gamma_1[0]).squeeze())
        g.maxflow()
        # Get the segments.
        sgm = g.get_grid_segments(nodeids) * 1

        # The labels should be 1 where sgm is False and 0 otherwise.
        new_gamma = np.int_(np.logical_not(sgm))
        # g.reset()
        self.gamma = new_gamma


    def update_u(self):
        # self.u = self.u + (self.gamma - (self.proba.data.numpy()>0.5)*1)
        self.u = self.u*0.1 + (self.gamma - self.proba.data.numpy())*0.000001
        # self.u = self.u + (self.gamma - self.proba.data.numpy())
        print(self.u.max())
        self.neighor*=1.01



    def update_s(self):
        a = 0.5 - (self.proba.data.numpy().squeeze() + self.v[0])
        original_shape = a.shape
        a_ = np.sort(a.ravel())
        useful_pixel_number = (a<0).sum()
        if self.lowerband< useful_pixel_number and self.upperband > useful_pixel_number:
            self.s = ((a<0)*1.0).reshape(original_shape)
        if useful_pixel_number < self.lowerband:
            self.s = ((a<=a_[self.lowerband])*1).reshape(original_shape)
        if useful_pixel_number > self.upperband:
            self.s = ((a<=a_[self.upperband])*1).reshape(original_shape)


    def update_v(self):
        # self.v = self.v + ((self.proba.data.numpy().squeeze()>0.5)*1.0-self.s.squeeze())
        self.v = self.v*0.99 + (self.proba.data.numpy().squeeze()- self.s.squeeze())*0.1
        pass

    #
    def update(self):

        # self.update_gamma()
        self.update_s()
        self.update_theta()

        self.update_v()

    def show_proba(self):
        proba= self.proba.data.numpy()[0]
        plt.figure(1)
        plt.clf()
        plt.imshow(proba,cmap='gray')
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.001)

    def show_gamma(self):
        plt.figure(2)
        plt.clf()
        plt.imshow(self.gamma,cmap='gray')
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.001)

    def show_s(self):
        plt.figure(2)
        plt.clf()
        plt.imshow(self.s,cmap='gray')
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.001)



    def show_u(self):
        plt.figure(3)
        plt.clf()
        plt.imshow(self.u[0],cmap='gray')
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.001)


    def show_v(self):
        plt.figure(3)
        plt.clf()
        plt.imshow(self.v[0],cmap='gray')
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.001)


if __name__ =="__main__":
    np.random.seed(1)
    # image_input = np.random.randn(*(100,100))
    # image_output = np.random.randn(*image_input.shape)
    #
    # net = Net(image_input.size)
    # output_image = net(torch.tensor(image_input).float().unsqueeze(0))
    # print()
    output_image = imread("PyMaxFlow_Examples/a2.png")/255.0
    f = lambda x,y: -x**2 + -y**2
    rng = 2
    x_ = np.linspace(-rng, rng, 64)
    y_ = np.linspace(-rng, rng, 64)
    X_, Y_ = np.meshgrid(x_, y_)
    Height = f(X_.reshape(-1), Y_.reshape(-1)).reshape(X_.shape)
    Height = Height- Height.min()
    Height = Height/Height.max()
    # plt.imshow(Height,cmap='gray')
    # plt.colorbar()
    # plt.show()
    # plt.contourf(X_, Y_, Height, 20, alpha=0.6, cmap=plt.cm.hot)
    # # 绘制等高线
    # C = plt.contour(X_, Y_, Height, 20, colors='black', linewidth=0.1)
    # # 显示各等高线的数据标签
    # plt.clabel(C, inline=True, fontsize=10)
    # plt.show()
    input_image =  np.random.randn(*output_image.shape)
    # net = train_pretrain_network(input_image,Height)
    # torch.save(net.state_dict(),'net_size.pth')
    net = Net(input_image.size)
    net.load_state_dict(torch.load('net_size.pth'))
    admm = ADMM(net,input_image,lower_band=4000,upper_band=5000)
    for i in range (10000):
        admm.update()
        admm.show_proba()
        # admm.show_gamma()
        admm.show_s()
        admm.show_v()

        print()