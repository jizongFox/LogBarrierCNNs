import torch
import torch.nn as nn
import torch.nn.functional as F

use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None,reduce=True,size_average=True):
        super().__init__()

        self.loss = nn.NLLLoss(weight,reduce=reduce,size_average=size_average)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs,dim=1), targets)

class partialCrossEntropyLoss2d(nn.Module):
    def __init__(self, weight,temperature = 1):
        super().__init__()
        self.temporature = temperature
        self.loss = nn.NLLLoss(weight, reduce=False, size_average=False)
    def forward(self, outputs, targets):
        elementwise_loss = self.loss(F.log_softmax(outputs/self.temporature,dim=1), targets)
        partialLossMask = targets.data.float()
        if partialLossMask.sum()==0:
            pass
        partialLoss  = (elementwise_loss * partialLossMask ).sum()/targets.data.sum().float()

        return partialLoss

class logBarrierLoss(nn.Module):

    def __init__(self,low_band,high_band,temporature=0.1):
        super().__init__()
        self.low_band = low_band
        self.high_band = high_band
        self.temperature = float(temporature)


    def forward(self, probability):
        total_pixel_number = list(probability.view(-1).shape)[0]
        if probability.sum(1).mean()!=1:
            probability= F.softmax(probability/self.temperature,dim=1)
        sum_pixels= probability[:,1,:,:].sum(-1).sum(-1)
        loss_table = []
        for image in sum_pixels:
            if image>= self.high_band:
                loss = (image-self.high_band)**2/total_pixel_number
            elif image <= self.low_band:
                loss = (sum_pixels - self.low_band) ** 2/total_pixel_number
            elif (image>self.low_band) and (image<self.high_band):
                loss = torch.tensor(0.0).float().to(device)
            else:
                raise ValueError
            loss_table.append(loss)
        loss_t = torch.stack(loss_table)
        return loss_t.mean()


def dice_loss(input, target):
    # with torch.no_grad:
    smooth = 1.

    iflat = input.view(input.size(0),-1)
    tflat = target.view(input.size(0),-1)
    intersection = (iflat * tflat).sum(1)
    # intersection = (iflat == tflat).sum(1)

    foreground_iou = float( ((2. * intersection + smooth).float() /  (iflat.sum(1) + tflat.sum(1) + smooth).float()).mean())
    # return ((2. * intersection + smooth).float() / (iflat.size(1)+ tflat.size(1) + smooth)).mean()


    iflat = 1- input.view(input.size(0),-1)
    tflat = 1- target.view(input.size(0),-1)
    intersection = (iflat * tflat).sum(1)
    # intersection = (iflat == tflat).sum(1)

    backgroud_iou = float( ((2. * intersection + smooth).float() /  (iflat.sum(1) + tflat.sum(1) + smooth).float()).mean())
    return [backgroud_iou, foreground_iou]

if __name__=="__main__":
    output_of_the_net = torch.rand(16, 2, 256,256)
    output_of_the_net = F.softmax(output_of_the_net, dim=1)
    criterion = logBarrierLoss(10,100)
    loss = criterion(output_of_the_net)
    print(loss)



