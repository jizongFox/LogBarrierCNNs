import sys,os
sys.path.extend([os.path.dirname(os.getcwd())])
from torch.utils.data import DataLoader
from torchvision import transforms
from Jizong_code import medicalDataLoader
from Jizong_code.network import UNet,SegNet
from Jizong_code.enet import Enet
import numpy as np,matplotlib.pyplot as plt
from Jizong_code.criterion import  CrossEntropyLoss2d,partialCrossEntropyLoss2d,logBarrierLoss,dice_loss
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from torchnet.meter import AverageValueMeter
from Jizong_code.utils import pred2segmentation,dice_loss,Colorize
from Jizong_code.visualize import Dashboard

board_image = Dashboard(server='http://turing.livia.etsmtl.ca',env="image")
board_loss = Dashboard(server='http://turing.livia.etsmtl.ca',env="loss")

cuda_device = "0"
batch_size = 1
batch_size_val = 1
num_workers = 16
lr = 0.0002
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
train_loader = DataLoader(train_set,batch_size=batch_size,num_workers=num_workers,shuffle=True,drop_last=False)

val_set = medicalDataLoader.MedicalImageDataset('val',root_dir,transform=transform,mask_transform=mask_transform,equalize=False)
val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)
num_classes=2
net = Enet(num_classes=num_classes).cuda()
# net.load_state_dict(torch.load('U_net_2Class.pth'))
# net.final = nn.Conv2d(64, 4, 1).cuda()
# net = Enet(num_classes=2).cuda()
optimiser = torch.optim.Adam(net.parameters(),lr=lr)
weight = torch.ones(num_classes)
# weight[0]=0
# criterion = CrossEntropyLoss2d(weight.cuda(),reduce=False, size_average=False).cuda()
partialCECriterion = partialCrossEntropyLoss2d(weight.cuda())
sizeCriterion = logBarrierLoss(0,1700)
global highest_dice_loss
highest_dice_loss =0


def val():
    global highest_dice_loss
    dice_loss_meter=AverageValueMeter()
    dice_loss_meter.reset()
    for i, (img, mask, weak_mask, _) in enumerate(val_loader):
        if (weak_mask.sum() <= 3) or (mask.sum() <=10):
            # print('No mask has been found')
            continue
        img, mask, weak_mask = img.cuda(), mask.cuda(),weak_mask.cuda()

        predict_ = F.softmax(net(img), dim=1)
        segm = pred2segmentation(predict_)
        diceloss= dice_loss(segm,mask)
        dice_loss_meter.add(diceloss.item())

        if i % 100 == 0:

            board_image.image(img[0], 'medical image')
            board_image.image(color_transform(weak_mask[0]), 'weak_mask')
            board_image.image(color_transform(segm[0]), 'prediction')
    board_loss.plot('dice_loss for validationset',dice_loss_meter.value()[0])

    if dice_loss_meter.value()[0]> highest_dice_loss:
        highest_dice_loss= dice_loss_meter.value()[0]
        torch.save(net.state_dict(), 'Enet_Square_barrier.pth')
        print('saved with dice:%f'%highest_dice_loss)


def train ():
    totalloss_meter = AverageValueMeter()
    sizeloss_meter = AverageValueMeter()
    celoss_meter = AverageValueMeter()

    for epoch in range(max_epoch):
        totalloss_meter.reset()
        celoss_meter.reset()
        sizeloss_meter.reset()
        if epoch %5==0:
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr * (0.9 ** (epoch // 3))
                print('learning rate:', param_group['lr'])
            print('save model:')
            # torch.save(net.state_dict(), 'U_net_2Class.pth')

        for i, (img,mask,weak_mask,_) in tqdm(enumerate(train_loader)):
            if (weak_mask.sum()==0) or (mask.sum()==0):
                # print('No mask has been found')
                continue
            img,mask,weak_mask=img.cuda(),mask.cuda(), weak_mask.cuda()
            optimiser.zero_grad()
            predict = net(img)
            loss_ce = partialCECriterion(predict,weak_mask.squeeze(1))
            # loss_ce = torch.Tensor([0]).cuda()
            # celoss_meter.add(loss_ce.item())
            loss_size = sizeCriterion(predict)
            # loss_size = torch.Tensor([0]).cuda()
            sizeloss_meter.add(loss_size.item())
            loss = loss_ce+loss_size
            totalloss_meter.add(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1e-4)
            optimiser.step()
        #     if i %50==0:
        #         predict_ = F.softmax(predict, dim=1)
        #         segm = pred2segmentation(predict)
        #         print("ce_loss:%.2f,  size_loss:%.2f ,FB percentage:%.2f"%(loss_ce.item(),loss_size.item(),((predict_[:, 1, :, :]*weak_mask.data.float()).sum()/weak_mask.data.float().sum()).item()))
        #         board_image.image(img[0],'medical image')
        #         board_image.image(color_transform(mask[0]),'weak_mask')
        #         board_image.image(color_transform(weak_mask[0]),'weak_mask')
        #         board_image.image(color_transform(segm[0]),'prediction')
        #         if totalloss_meter.value()[0] < 1:
        #             board_loss.plot('ce_loss', -np.log(loss_ce.item()+1e-6))
        #             board_loss.plot('size_loss', -np.log(loss_size.item()+1e-6))
        #             # board_loss.plot('size_loss', -np.log(sizeloss_meter.value()[0]))
        # # print('train loss:%.5f'%celoss_meter.value()[0])
        val()








if __name__=="__main__":
    train()





