# -*- coding: utf-8 -*-
import numpy as np,pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.misc import imread


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
        output = self.linear4(output)

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
    def __init__(self,neural_network, input_image, lower_band,upper_band) -> None:
        super().__init__()
        self.neural_net = neural_network,
        self.lowerband= lower_band
        self.upperband = upper_band
        self.learning_rate = 1e-3
        self.opitimiser = torch.optim.Adam(self.neural_net.parameters(),lr = self.learning_rate)
        self.input_image = input_image
    def forward_image(self):
        self.output_image = self.neural_net(self.input_image)

    def heatmap2segmentation(self, heatmap):
        return heatmap.max(1)[1]

    def update_theta(self):
        pass

    def update_gamma(self):
        pass

    def update_u(self):
        pass

    def update(self):

        self.update_theta()
        self.update_gamma()
        self.update_u()





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
    plt.imshow(net(torch.tensor(input_image).float().unsqueeze(0)).data.numpy().squeeze())
    plt.show()
