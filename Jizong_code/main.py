import os,pandas as pd
import sys

sys.path.extend([os.path.dirname(os.getcwd())])
from torch.utils.data import DataLoader
from torchvision import transforms
import medicalDataLoader
from enet import Enet
import numpy as np
from criterion import partialCrossEntropyLoss2d, logBarrierLoss, dice_loss
import torch, torch.nn.functional as F
from tqdm import tqdm
from torchnet.meter import AverageValueMeter
from utils import pred2segmentation, dice_loss, Colorize
from visualize import Dashboard
import logging
from torchvision.utils import save_image,make_grid
import click

# board_train_image = Dashboard(server='http://pascal.livia.etsmtl.ca', env="train_image")
# board_val_image = Dashboard(server='http://pascal.livia.etsmtl.ca', env="vali_image")
# board_loss = Dashboard(server='http://pascal.livia.etsmtl.ca', env="loss")
# board_local = Dashboard(server='http://localhost', env="val")
use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')

batch_size = 4
batch_size_val = 1
num_workers = 4
lr = 0.0001
max_epoch = 200
# root_dir = '/home/AN96120/pythonProject/newidea/DGA1032/dataset/ACDC-2D-All'
root_dir ='../ACDC-2D-All'
model_dir = 'model'

color_transform = Colorize()
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])
mask_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])

train_set = medicalDataLoader.MedicalImageDataset('train', root_dir, transform=transform, mask_transform=mask_transform,
                                                  augment=True, equalize=False)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=False)

val_set = medicalDataLoader.MedicalImageDataset('val', root_dir, transform=transform, mask_transform=mask_transform,
                                                equalize=False)
val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)
num_classes = 2
net = Enet(num_classes=num_classes).to(device)

# net.load_state_dict(torch.load('Enet_Square_barrier_0.776037_fiou_0.557611.pth', map_location=lambda storage, loc: storage))


optimiser = torch.optim.Adam(net.parameters(), lr=lr)
weight = torch.ones(num_classes)
weight[0] = 0
# criterion = CrossEntropyLoss2d(weight.to(device),reduce=False, size_average=False).to(device)
partialCECriterion = partialCrossEntropyLoss2d(weight.to(device))

sizeCriterion = logBarrierLoss(97.9, 1722.6)

highest_dice_loss = -1

def val_for_local():
    dice_loss_meter = AverageValueMeter()
    fiou_loss_meter = AverageValueMeter()
    biou_loss_meter = AverageValueMeter()
    dice_loss_meter.reset()
    fiou_loss_meter.reset()
    biou_loss_meter.reset()
    net.eval()
    if not os.path.isdir('results'):
        os.mkdir('results')

    for i, (img, mask, weak_mask, pathname) in enumerate(val_loader):
        images = []
        basename = os.path.basename(pathname[0])
        if (weak_mask.sum() <= 1) or (mask.sum() <= 1):
            continue
        img, mask, weak_mask = img.to(device), mask.to(device), weak_mask.to(device)

        predict_ = F.softmax(net(img), dim=1)
        segm = pred2segmentation(predict_)
        [diceloss_B, diceloss_F] = dice_loss(segm, mask)
        dice_loss_meter.add((diceloss_F + diceloss_B) / 2.0)
        fiou_loss_meter.add(diceloss_F)
        biou_loss_meter.add(diceloss_B)

        # board_local.image(img[0], 'medical image')
        # board_local.image(color_transform(weak_mask[0]), 'weak_mask')
        # board_local.image(color_transform(segm[0]), 'prediction')
        images.extend([img[0],weak_mask[0].float(),mask[0].float(),segm.float()])
        grid = make_grid(images)
        save_image(grid,'results/fiou_%.2f_with_%s.png'%(diceloss_F,basename))


    # board_loss.plot('mean dice_loss for validationset', dice_loss_meter.value()[0])
    # board_loss.plot('foreground dice loss for val', fiou_loss_meter.value()[0])
    net.train()
    return [biou_loss_meter.value()[0], fiou_loss_meter.value()[0]]



def evaluate(dataloader):
    global highest_dice_loss
    dice_loss_meter = AverageValueMeter()
    fiou_loss_meter = AverageValueMeter()
    biou_loss_meter = AverageValueMeter()
    dice_loss_meter.reset()
    fiou_loss_meter.reset()
    biou_loss_meter.reset()
    net.eval()
    for i, (img, mask, weak_mask, _) in enumerate(dataloader):
        if (weak_mask.sum() <= 1) or (mask.sum() <= 1):
            continue
        img, mask, weak_mask = img.to(device), mask.to(device), weak_mask.to(device)

        predict_ = F.softmax(net(img), dim=1)
        segm = pred2segmentation(predict_)
        [diceloss_B, diceloss_F] = dice_loss(segm, mask)
        dice_loss_meter.add((diceloss_F + diceloss_B) / 2.0)
        fiou_loss_meter.add(diceloss_F)
        biou_loss_meter.add(diceloss_B)

    #     if i % 30 == 0:
    #         board_val_image.image(img[0], 'medical image')
    #         board_val_image.image(color_transform(weak_mask[0]), 'weak_mask')
    #         board_val_image.image(color_transform(segm[0]), 'prediction')
    # board_loss.plot('mean dice_loss for validationset', dice_loss_meter.value()[0])
    # board_loss.plot('foreground dice loss for val', fiou_loss_meter.value()[0])


    if dice_loss_meter.value()[0] > highest_dice_loss:
        highest_dice_loss = dice_loss_meter.value()[0]
        torch.save(net.state_dict(), 'Enet_Square_barrier_%.6f_fiou_%.6f.pth'%(highest_dice_loss,fiou_loss_meter.value()[0]))
        print('saved with dice:%f' % highest_dice_loss)
    net.train()
    return [biou_loss_meter.value()[0], fiou_loss_meter.value()[0]]

@click.command()
@click.option('--lamda',default=1.0,help='weights between the ce and size loss')
def train(lamda):
    iou_tables = []
    totalloss_meter = AverageValueMeter()
    sizeloss_meter = AverageValueMeter()
    celoss_meter = AverageValueMeter()


    for epoch in range(max_epoch):
        totalloss_meter.reset()
        celoss_meter.reset()
        sizeloss_meter.reset()
        if epoch % 5 == 0:
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr * (0.95 ** (epoch // 3))
                print('learning rate:', param_group['lr'])
            print('save model:')
            # torch.save(net.state_dict(), 'U_net_2Class.pth')

        for i, (img, mask, weak_mask, _) in tqdm(enumerate(train_loader)):
            if (weak_mask.sum() == 0) or (mask.sum() <= 5):
                # print('No mask has been found')
                continue

            img, mask, weak_mask = img.to(device), mask.to(device), weak_mask.to(device)
            optimiser.zero_grad()
            predict = net(img)
            loss_ce = partialCECriterion(predict, weak_mask.squeeze(1))
            # loss_ce = torch.Tensor([0]).to(device)
            # celoss_meter.add(loss_ce.item())
            loss_size = sizeCriterion(predict)
            # loss_size = torch.Tensor([0]).to(device)
            sizeloss_meter.add(loss_size.item())
            loss = loss_ce + loss_size *lamda
            totalloss_meter.add(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1e-4)
            optimiser.step()
        #     if i % 50 == 0:
        #         net.eval()
        #         predict = net(img)
        #         predict_ = F.softmax(predict, dim=1)
        #         segm = pred2segmentation(predict)
        #         print("ce_loss:%.4f,  size_loss:%.4f, FB percentage:%.2f" % (loss_ce.item(), loss_size.item(),\
        #                                                                      ((predict_[:,1:,:,:] * weak_mask.data.float()).sum() / weak_mask.data.float().sum()).item()))
        #         board_train_image.image(img[0], 'medical image')
        #         board_train_image.image(color_transform(mask[0]), 'weak_mask')
        #         board_train_image.image(color_transform(weak_mask[0]), 'weak_mask')
        #         board_train_image.image(color_transform(segm[0]), 'prediction')
        #         if totalloss_meter.value()[0] < 1:
        #             board_loss.plot('ce_loss', loss_ce.item())
        #             board_loss.plot('size_loss', loss_size.item())
        #             # board_loss.plot('size_loss', -np.log(sizeloss_meter.value()[0]))
        # # print('train loss:%.5f'%celoss_meter.value()[0])
        #         net.train()
        train_ious=evaluate(train_loader)
        val_ious = evaluate(val_loader)
        ious = np.array([train_ious,val_ious]).ravel().tolist()
        iou_tables.append(ious)
        try:
            pd.DataFrame(iou_tables,columns=['train_background','train_foreground','val_background','val_foreground']).to_csv('ious_%.2f.csv'%lamda)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    train()
