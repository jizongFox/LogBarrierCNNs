import sys,os
sys.path.extend([os.path.dirname(os.getcwd())])
from torch.utils.data import DataLoader
from torchvision import transforms
from Jizong_multi_local_expert import medicalDataLoader
from Jizong_multi_local_expert.network import UNet,SegNet
from Jizong_multi_local_expert.enet import Enet
import numpy as np,matplotlib.pyplot as plt
from Jizong_multi_local_expert.criterion import  CrossEntropyLoss2d
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from torchnet.meter import AverageValueMeter
from Jizong_multi_local_expert.utils import pred2segmentation,dice_loss,Colorize
from Jizong_multi_local_expert.visualize import Dashboard
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--pretrained_model', type=int,default=1)



args = parser.parse_args()
pretrained_model = args.pretrained_model
board = Dashboard(env='subclass%d'%pretrained_model)
cuda_device = "%d"%pretrained_model
batch_size = 10
batch_size_val = 10
num_workers = 4
lr = 0.001
max_epoch = 100
root_dir = '../ACDC-2D-All'
# model_dir = 'model'
num_classes=2

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
color_transform = Colorize()
transform = transforms.Compose([
    transforms.ToTensor()
])
mask_transform = transforms.Compose([
    transforms.ToTensor()
])
train_set_1 = medicalDataLoader.MedicalImageDataset_LocalExpert('train',root_dir,size_range=(0,200),transform=transform,mask_transform=mask_transform,augment=False,equalize=False)
train_loader_1 = DataLoader(train_set_1,batch_size=batch_size,num_workers=num_workers,shuffle=True)

train_set_2 = medicalDataLoader.MedicalImageDataset_LocalExpert('train',root_dir,size_range=(201,1000),transform=transform,mask_transform=mask_transform,augment=False,equalize=False)
train_loader_2 = DataLoader(train_set_2,batch_size=batch_size,num_workers=num_workers,shuffle=True)

train_set_3 = medicalDataLoader.MedicalImageDataset_LocalExpert('train',root_dir,size_range=(1001,10000),transform=transform,mask_transform=mask_transform,augment=False,equalize=False)
train_loader_3 = DataLoader(train_set_3,batch_size=batch_size,num_workers=num_workers,shuffle=True)

# create 3 networks and 3 optimiser
for i in range (1,4):
    exec('net'+str(i)+ '= UNet(num_classes=2).cuda()')
    exec('optimiser'+str(i)+'=torch.optim.Adam(net'+str(i)+'.parameters(),lr=lr,weight_decay=1e-6)')
weight = torch.ones(num_classes)
# weight[0]=0
criterion = CrossEntropyLoss2d(weight.cuda()).cuda()

def pretrain_i_network(i):
    for j, (img, mask, _, _) in tqdm(enumerate(eval('train_loader_'+str(i)))):
        img, mask = img.cuda(), mask.cuda()
        eval('optimiser'+str(i)).zero_grad()
        predict = eval('net'+str(i))(img)
        loss = criterion(predict, mask.squeeze(1))
        segm = pred2segmentation(predict)
        loss.backward()
        eval('optimiser'+str(i)).step()


def __test__val__data(i):

    def __predict_and_show_(img,mask,):
        img, mask = img.cuda(), mask.cuda()
        predict =  eval('net'+str(i))(img)
        segm = pred2segmentation(predict)
        board.image(img[0],'medical image')
        board.image(color_transform(mask[0]),'mask')
        board.image(color_transform(segm[0]),'prediction')


    for j, ((img1, mask1, _, _), (img2, mask2, _, _), (img3, mask3, _, _)) in tqdm(enumerate(zip(train_loader_1, train_loader_2, train_loader_3))):
        __predict_and_show_(img1,mask1)
        __predict_and_show_(img2,mask2)
        __predict_and_show_(img3, mask3)
        torch.save(eval('net'+str(i)).state_dict(), 'subclass%d.pth'%pretrained_model)
        print('subclass%d.pth'%pretrained_model,' saved')
        break


for i in range (1000):
    pretrain_i_network(pretrained_model)
    if (i+1)%10==0:
        __test__val__data(pretrained_model)















'''

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




'''


