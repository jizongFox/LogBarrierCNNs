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
    for i in range(50):
        optimiser.zero_grad()
        output = net(input_image)
        loss = criterion(output, output_image)
        loss.backward()
        print(loss.item())
        optimiser.step()
    return net

class ADMM(object):
    '''
    min = R(gamma)  st: gamma = proba

    augmented lagrange multiplier:
    L = lambda R(gamma) + sum: u (gamma - proba) + p/2|gamma - proba|2^2

    in a scaled form:

    L = lambda R(gamma) + ur |gamma - prob + u|2^2

    to update theta:

    argmin: l_theta = ur |gamma - prob +u|2^2

    to update gamma:

    argmin: l_gamma = lambda R(gamma) _(unary term) + pixelwise sum of  (gamma + 2 gamma(u - prob) + (u - prob)^2)
                    = lambda R(gamma) + pixelwised sum of 2 gamma(1/2 + u - prob)

                    this can be extracted as an unary term of penalty (y==1)= 1/2 +u - prob and penalty(y==0) = 0
                    and lambda can be viewed as the boundary term .

    to update the multipiler
    u = u + gamma - prob


    '''
    def __init__(self,neural_network, input_image, lower_band,upper_band) -> None:
        super().__init__()
        self.reset()
        self.neural_net = neural_network
        self.lowerband= lower_band
        self.upperband = upper_band
        self.learning_rate = 1e-4
        self.opitimiser = torch.optim.SGD(self.neural_net.parameters(),lr = self.learning_rate,momentum=0.9,nesterov=True)
        self.input_image = input_image
        self.forward_image()
        self.p = 1
        self.neighor=0.05

    def reset(self):
        self.input_image=None
        self.proba=None
        self.gamma = None
        self.p =1


    def forward_image(self):
        self.proba = self.neural_net(torch.tensor(self.input_image).float().unsqueeze(0))
        self.gamma = np.zeros((64,64))
        self.u = np.zeros((64,64))

    def heatmap2segmentation(self, heatmap):
        return heatmap.max(1)[1]

    def update_theta(self):
        for i in range(1):
            self.opitimiser.zero_grad()
            loss =self.p*(torch.tensor(self.gamma +self.u).float() - self.proba).norm(2)**2
            loss.backward()
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
        self.u = self.u*0.9 + (self.gamma - (self.proba.data.numpy()>0.5)*1)*0.1
    #
    def update(self):

        self.update_theta()
        self.update_gamma()
        self.update_u()

    def show_proba(self):
        proba= self.proba.data.numpy()[0]
        plt.figure(1)
        plt.imshow(proba,cmap='gray')
        plt.show(block=False)
        plt.pause(0.001)

    def show_gamma(self):
        plt.figure(2)
        plt.imshow(self.gamma,cmap='gray')
        plt.show(block=False)
        plt.pause(0.001)
    def show_u(self):
        plt.figure(3)
        plt.imshow(self.u[0],cmap='gray')
        plt.show(block=False)
        plt.pause(0.001)

if __name__ =="__main__":
    # image_input = np.random.randn(*(100,100))
    # image_output = np.random.randn(*image_input.shape)
    #
    # net = Net(image_input.size)
    # output_image = net(torch.tensor(image_input).float().unsqueeze(0))
    # print()
    output_image = imread("PyMaxFlow_Examples/a2.png")/255.0
    input_image =  np.random.randn(*output_image.shape)
    net = train_pretrain_network(input_image,output_image)

    admm = ADMM(net,input_image,lower_band=5,upper_band=10)
    for i in range (100):
        admm.update()
        admm.show_proba()
        admm.show_gamma()
        admm.show_u()
        print()