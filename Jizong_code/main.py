import sys,os
sys.path.extend([os.path.dirname(os.getcwd())])
from torch.utils.data import DataLoader
from torchvision import transforms
from Jizong_code import medicalDataLoader
from Jizong_code.network import UNet,SegNet
from Jizong_code.enet import Enet
import numpy as np,matplotlib.pyplot as plt
from Jizong_code.criterion import  CrossEntropyLoss2d
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from torchnet.meter import AverageValueMeter
from Jizong_code.utils import pred2segmentation,dice_loss,Colorize
from Jizong_code.visualize import Dashboard

board = Dashboard()
cuda_device = "0"
batch_size = 4
batch_size_val = 4
num_workers = 4
lr = 0.001
max_epoch = 100
root_dir = '../ACDC-2D-All'
model_dir = 'model'

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
color_transform = Colorize()
transform = transforms.Compose([
    transforms.ToTensor()
])
mask_transform = transforms.Compose([
    transforms.ToTensor()
])
train_set = medicalDataLoader.MedicalImageDataset('train',root_dir,transform=transform,mask_transform=mask_transform,augment=True,equalize=False)
train_loader = DataLoader(train_set,batch_size=batch_size,num_workers=num_workers,shuffle=True)

val_set = medicalDataLoader.MedicalImageDataset('val',root_dir,transform=transform,mask_transform=mask_transform,equalize=False)
val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)
num_classes=2
net = UNet(num_classes=num_classes).cuda()
net.load_state_dict(torch.load('U_net_2Class.pth'))
# net.final = nn.Conv2d(64, 4, 1).cuda()
# net = Enet(num_classes=2).cuda()
optimiser = torch.optim.Adam(net.parameters(),lr=lr)
weight = torch.ones(num_classes)
# weight[0]=0
criterion = CrossEntropyLoss2d(weight.cuda()).cuda()

if __name__=="__main__":
    celoss_meter = AverageValueMeter()

    for epoch in range(max_epoch):
        celoss_meter.reset()
        if epoch %5==0:
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr * (0.98 ** (epoch // 3))
                print('learning rate:', param_group['lr'])
            print('save model:')
            torch.save(net.state_dict(), 'U_net_2Class.pth')

        for i, (img,mask,_,_) in tqdm(enumerate(train_loader)):
            img,mask=img.cuda(),mask.cuda()
            optimiser.zero_grad()
            predict = net(img)
            loss = criterion(predict,mask.squeeze(1))
            segm = pred2segmentation(predict)
            loss.backward()
            optimiser.step()
            celoss_meter.add(loss.item())
            # if i %100==0:
            #     board.image(img[0],'medical image')
            #     board.image(color_transform(mask[0]),'mask')
            #     board.image(color_transform(segm[0]),'prediction')
        print('train loss:%.5f'%celoss_meter.value()[0])



        for i, (img, mask, _, _) in enumerate(val_loader):

            img, mask = img.cuda(), mask.cuda()
            predict_ = F.softmax(net(img),dim=1)
            segm = pred2segmentation(predict_)
            if i %100==0:
                board.image(img[0],'medical image')
                board.image(color_transform(mask[0]),'mask')
                board.image(color_transform(segm[0]),'prediction')







