import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import torch
from torch import nn
from torchvision import models

import pdb

class FCN8s(nn.Module):
    def __init__(self, n_classes=21):
        super(FCN8s, self).__init__()
        print('~'*50)
        print(' ----- Creating FCN8s network...')
        print('~'*50)

        self.n_classes = n_classes

        print(' Loading pre-trained model...')
        #vgg = models.vgg16(pretrained=True)

        #features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        
        #features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        '''self.convBlock1 = nn.Sequential(*features[0:5])
        self.convBlock2 = nn.Sequential(*features[5:10])
        self.convBlock3 = nn.Sequential(*features[10:17])
        self.convBlock4 = nn.Sequential(*features[17:24])
        self.convBlock5 = nn.Sequential(*features[24:31])'''

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.convBlock3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.convBlock4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.convBlock5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)
            

        # To freeze the learned filters
        '''for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad = False'''

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),)

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

    def forward(self, x):
        conv1 = self.convBlock1(x)
        conv2 = self.convBlock2(conv1)
        conv3 = self.convBlock3(conv2)
        conv4 = self.convBlock4(conv3)
        conv5 = self.convBlock5(conv4)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)

        score = F.upsample_bilinear(score, score_pool4.size()[2:])
        score += score_pool4
        score = F.upsample_bilinear(score, score_pool3.size()[2:])
        score += score_pool3
        out = F.upsample_bilinear(score, x.size()[2:])

        return out
