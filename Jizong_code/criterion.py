import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self,low_band,high_band,):
        super().__init__()
        self.low_band = low_band
        self.high_band = high_band


    def forward(self, probability):
        if probability.sum(1).mean()!=1:
            probability= F.softmax(probability,dim=1)
        sum_pixels= probability[:,1,:,:].sum(-1).sum(-1)
        loss_t = torch.Tensor([0]).cuda()
        for image in sum_pixels:
            if image>= self.high_band:
                loss = (image-self.high_band)**2
            if image <= self.low_band:
                loss = (sum_pixels - self.low_band) ** 2
            if (image>self.low_band) and (image<self.high_band):
                loss = torch.Tensor([0]).cuda()
            try:
                loss_t=torch.cat((loss_t,loss.unsqueeze(0)),0)
            except:
                loss_t = torch.cat((loss_t, loss), 0)
        return loss_t.sum()/(loss_t.shape[0]-1)


def dice_loss(input, target):
    # with torch.no_grad:
    smooth = 1.

    iflat = input.view(input.size(0),-1)
    tflat = target.view(input.size(0),-1)
    intersection = (iflat * tflat).sum(1)
    # intersection = (iflat == tflat).sum(1)

    return ((2. * intersection + smooth).float() /  (iflat.sum(1) + tflat.sum(1) + smooth).float()).mean()
    # return ((2. * intersection + smooth).float() / (iflat.size(1)+ tflat.size(1) + smooth)).mean()

if __name__=="__main__":
    output_of_the_net = torch.rand(16, 2, 256,256)
    output_of_the_net = F.softmax(output_of_the_net, dim=1)
    criterion = logBarrierLoss(10,100)
    loss = criterion(output_of_the_net)
    print(loss)



