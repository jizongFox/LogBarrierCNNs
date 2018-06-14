import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
import skimage.transform as skiTransf
from progressBar import printProgressBar
import scipy.io as sio
import pdb


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class computeDiceOneHotBinary(nn.Module):
    def __init__(self):
        super(computeDiceOneHotBinary, self).__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-8) / (sum + 1e-8)

        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def inter(self, input, target):
        return (input * target).float().sum()

    def sum(self, input, target):
        return input.sum() + target.sum()

    def forward(self, pred, GT):
        # GT is 4x320x320 of 0 and 1
        # pred is converted to 0 and 1
        batchsize = GT.size(0)
        DiceN = to_var(torch.zeros(batchsize, 2))
        DiceB = to_var(torch.zeros(batchsize, 2))
        #DiceW = to_var(torch.zeros(batchsize, 2))
        #DiceT = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            DiceN[i, 0] = self.inter(pred[i, 0], GT[i, 0])
            DiceB[i, 0] = self.inter(pred[i, 1], GT[i, 1])
            #DiceW[i, 0] = self.inter(pred[i, 2], GT[i, 2])
            #DiceT[i, 0] = self.inter(pred[i, 3], GT[i, 3])

            DiceN[i, 1] = self.sum(pred[i, 0], GT[i, 0])
            DiceB[i, 1] = self.sum(pred[i, 1], GT[i, 1])
            #DiceW[i, 1] = self.sum(pred[i, 2], GT[i, 2])
            #DiceT[i, 1] = self.sum(pred[i, 3], GT[i, 3])

        return DiceN, DiceB #, DiceW, DiceT
        
        
class computeDiceOneHot(nn.Module):
    def __init__(self):
        super(computeDiceOneHot, self).__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-8) / (sum + 1e-8)

        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def inter(self, input, target):
        return (input * target).float().sum()

    def sum(self, input, target):
        return input.sum() + target.sum()

    def forward(self, pred, GT):
        # GT is 4x320x320 of 0 and 1
        # pred is converted to 0 and 1
        batchsize = GT.size(0)
        DiceN = to_var(torch.zeros(batchsize, 2))
        DiceB = to_var(torch.zeros(batchsize, 2))
        DiceW = to_var(torch.zeros(batchsize, 2))
        DiceT = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            DiceN[i, 0] = self.inter(pred[i, 0], GT[i, 0])
            DiceB[i, 0] = self.inter(pred[i, 1], GT[i, 1])
            DiceW[i, 0] = self.inter(pred[i, 2], GT[i, 2])
            DiceT[i, 0] = self.inter(pred[i, 3], GT[i, 3])

            DiceN[i, 1] = self.sum(pred[i, 0], GT[i, 0])
            DiceB[i, 1] = self.sum(pred[i, 1], GT[i, 1])
            DiceW[i, 1] = self.sum(pred[i, 2], GT[i, 2])
            DiceT[i, 1] = self.sum(pred[i, 3], GT[i, 3])

        return DiceN, DiceB , DiceW, DiceT


def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)



def getSingleImageBin(pred):
    # input is a 4-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    Val = to_var(torch.zeros(2))
    Val[1] = 1.0
    
    x = predToSegmentation(pred)

    out = x * Val.view(1, 2, 1, 1)
    return out.sum(dim=1, keepdim=True)
    
def getSingleImage(pred):
    # input is a 4-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    Val = to_var(torch.zeros(4))
    Val[1] = 0.33333334
    Val[2] = 0.66666669
    Val[3] = 1.0
    
    x = predToSegmentation(pred)

    out = x * Val.view(1, 4, 1, 1)
    return out.sum(dim=1, keepdim=True)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def getOneHotTumorClass(batch):
    data = batch.cpu().data.numpy()
    classLabels = np.zeros((data.shape[0], 2))

    tumorVal = 1.0
    for i in range(data.shape[0]):
        img = data[i, :, :, :]
        values = np.unique(img)
        if len(values) > 3:
            classLabels[i, 1] = 1
        else:
            classLabels[i, 0] = 1

    tensorClass = torch.from_numpy(classLabels).float()

    return Variable(tensorClass.cuda())


def getOneHotSegmentation(batch):
    backgroundVal = 0
    label1 = 0.33333334
    label2 = 0.66666669
    label3 = 1.0
   
    oneHotLabels = torch.cat((batch == backgroundVal, batch == label1, batch == label2, batch == label3),
                             dim=1)
    return oneHotLabels.float()


def getOneHotSegmentation_Bin(batch):
    backgroundVal = 0
    label1 = 1.0
   
    oneHotLabels = torch.cat((batch == backgroundVal, batch == label1),  dim=1)
    return oneHotLabels.float()
    
def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.3137255, 0.627451 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3
    spineLabel = 0.33333334
    return (batch / spineLabel).round().long().squeeze()

from scipy import ndimage

def getValuesRegression(image):
    label1 = 0.33333334
    
    feats = np.zeros((image.shape[0],3))

    for i in range(image.shape[0]):
        imgT = image[i,0,:,:].numpy()
        idx = np.where(imgT==label1)
        img = np.zeros(imgT.shape)
        img[idx]=1
        sizeRV = len(idx[0])
        [x,y] = ndimage.measurements.center_of_mass(img)
        
        if sizeRV == 0:
            x = 0
            y = 0
            
        feats[i,0] = sizeRV
        feats[i,1] = x
        feats[i,2] = y
        
        
        #print(' s: {}, x: {}, y: {} '.format(sizeRV,x,y))
        
    return feats
    

def saveImagesSegmentation(net, img_batch, batch_size, epoch, modelName, deepSupervision):
    # print(" Saving images.....")
    path = 'Results/' + modelName
    if not os.path.exists(path):
        os.makedirs(path)
    total = len(img_batch)
    net.eval()
    softMax = nn.Softmax()
    softMax.cuda()
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="Saving images.....", length=30)
        image, labels, weakly_labels, img_names = data
        
        MRI = to_var(image)
        Segmentation = to_var(labels)
            
        if deepSupervision == False:
            # No deep supervision
            segmentation_prediction = net(MRI)
        else:
            # Deep supervision
            segmentation_prediction, seg_3, seg_2, seg_1 = net(MRI)

        pred_y = softMax(segmentation_prediction)
        #segmentation = getSingleImage(segmentation_prediction)
        #segmentation = getSingleImageBin(segmentation_prediction)
        segmentation = getSingleImageBin(pred_y)
        
        
        out = torch.cat((MRI, segmentation, Segmentation))
        
        name2save = img_names[0].split('./ACDC-2D-All/val/Img/')
        
        torchvision.utils.save_image(segmentation.data, os.path.join(path, name2save[1]))
        
    printProgressBar(total, total, done="Images saved !")
        
def saveImages(net, img_batch, batch_size, epoch, modelName, deepSupervision):
    # print(" Saving images.....")
    path = 'Results/' + modelName
    if not os.path.exists(path):
        os.makedirs(path)
    total = len(img_batch)
    net.eval()
    softMax = nn.Softmax()
    softMax.cuda()
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="Saving images.....", length=30)
        image, labels, weakly_labels, img_names = data

        MRI = to_var(image)
        Segmentation = to_var(labels)
            
        if deepSupervision == False:
            # No deep supervision
            segmentation_prediction = net(MRI)
        else:
            # Deep supervision
            segmentation_prediction, seg_3, seg_2, seg_1 = net(MRI)

        temperature = 0.1
        pred_y = softMax(segmentation_prediction/temperature)
            
        segmentation = getSingleImageBin(pred_y)
        
        out = torch.cat((MRI, segmentation, Segmentation))

        torchvision.utils.save_image(segmentation.data, os.path.join(path, str(i) + '_Ep_' + str(epoch) + '.png'), normalize=False, range=None, scale_each=False)
        
        '''torchvision.utils.save_image(out.data, os.path.join(path, str(i) + '_Ep_' + str(epoch) + '.png'),
                                     nrow=batch_size,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False,
                                     pad_value=0)'''
    printProgressBar(total, total, done="Images saved !")

def saveImagesAsMatlab(net, img_batch, batch_size, epoch):
    print(" Saving images.....")
    path = 'ResultsMatlab-ENet-ACDC-York'
    if not os.path.exists(path):
        os.makedirs(path)
    total = len(img_batch)
    net.eval()
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="Saving images.....", length=30)
        image, labels, img_names = data

        MRI = to_var(image)
        Segmentation = to_var(labels)

        segmentation_prediction = net(MRI)

        segmentation = getSingleImage(segmentation_prediction)
        nameT = img_names[0].split('Img/')
        nameT = nameT[1].split('.png')
        pred =  segmentation.data.cpu().numpy().reshape(256,256)
        sio.savemat(os.path.join(path, nameT[0] + '.mat'), {'pred':pred})
      
    printProgressBar(total, total, done="Images saved !")
    
def inference(net, temperature, img_batch, batch_size, epoch, deepSupervision, modelName, minSize, maxSize):

    directory = 'Results/ImagesViolationConstraint/NIPS/' + modelName +'/Epoch_' + str(epoch)
    if not os.path.exists(directory):
        os.makedirs(directory)
            
    total = len(img_batch)

    Dice1 = torch.zeros(total, 2)
    
    net.eval()
    
    img_names_ALL = []

    softMax = nn.Softmax()
    softMax.cuda()
    
    dice = computeDiceOneHotBinary().cuda()

    targetSizeArrBatches_val = []
    predSizeArrBatches_val = []
    violatedCases = []
    numImages = len(img_batch)

    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, labels_weak, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)
        Segmentation = to_var(labels)

        
        if deepSupervision == False:
            segmentation_prediction = net(MRI)
        else:
            segmentation_prediction, seg3,seg2,seg1 = net(MRI)

        Segmentation_planes = getOneHotSegmentation_Bin(Segmentation)
        
        pred_y = softMax(segmentation_prediction/temperature) 

        # ----- To compute the predicted and target size ------
        predSize = torch.sum((pred_y[:,1,:,:] > 0.5).type(torch.FloatTensor))
        predSizeNumpy = predSize.cpu().data.numpy()[0]

        LV_target = (labels == 1).type(torch.FloatTensor)
        targetSize = torch.sum(LV_target)
        targetSizeNumpy = targetSize # targetSize.cpu().data.numpy()[0]

        predSizeArrBatches_val.append(predSizeNumpy)
        targetSizeArrBatches_val.append(targetSizeNumpy)
            
        '''softmax_y = pred_y.cpu().data.numpy()
        softB = softmax_y[:,1,:,:]

        # Hard-Size
        pixelsClassB = np.where( softB > 0.5 )
        
        predLabTemp = np.zeros(softB.shape)
        predLabTemp[pixelsClassB] = 1.0
        sizePredNumpy = predLabTemp.sum()
        #minSize = 97.9
        #maxSize = 1722.6
        
        sizeCase = []
        idx = np.where(labels.numpy()==1.0)
        sizeLV_GT = len(idx[0])
        sizesGT.append(sizeLV_GT)
        sizesPred.append(sizePredNumpy)'''

        '''if sizeLV_GT > 0:
            if sizePredNumpy < minSize:
                #violated +=1
                out = torch.cat((MRI, pred_y[:,1,:,:].view(1,1,256,256), Segmentation))
                name2save = img_names[0].split('./ACDC-2D-All/val/Img/')
                name2save = name2save[1].split('.png')
                torchvision.utils.save_image(out.data, os.path.join(directory, name2save[0]+'_Lower_'+str(minSize-sizePredNumpy)+'.png'),
                                     nrow=batch_size,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False,
                                     pad_value=0)
            
            if sizePredNumpy > maxSize:
                #violated +=1
                
                out = torch.cat((MRI, pred_y[:,1,:,:].view(1,1,256,256), Segmentation))
                name2save = img_names[0].split('./ACDC-2D-All/val/Img/')
                name2save = name2save[1].split('.png')
                torchvision.utils.save_image(out.data, os.path.join(directory, name2save[0]+'_Upper_'+str(sizePredNumpy-maxSize)+'.png'),
                                     nrow=batch_size,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False,
                                     pad_value=0)

        else:
            if sizePredNumpy > 0:
                a = 0
                #segmentation = getSingleImageBin(pred_y)
                out = torch.cat((MRI, pred_y[:,1,:,:].view(1,1,256,256), Segmentation))
                name2save = img_names[0].split('./ACDC-2D-All/val/Img/')

                name2save = name2save[1].split('.png')
                torchvision.utils.save_image(out.data, os.path.join(directory, name2save[0]+'_'+str(sizePredNumpy)+'.png'),
                                     nrow=batch_size,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False,
                                     pad_value=0)'''

               
        #DicesN, Dices1 = dice(segmentation_prediction, Segmentation_planes)
        DicesN, Dices1 = dice(pred_y, Segmentation_planes)

        Dice1[i] = Dices1.data

        # Save images
        # directory = 'resultBladder'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # filenameImg = os.path.join(directory, "original_image_{}_{}.png".format(epoch, i))
        # filenameMask = os.path.join(directory, "groundTruth_image_{}_{}.png".format(epoch, i))
        # filenamePred = os.path.join(directory, "Prediction_{}_{}.png".format(epoch, i))
    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    ValDice1 = DicesToDice(Dice1)
   
    #percViol = (violated/numImages)*100
    
    #print(' [VAL] Constrained violated in {} images: {:.4f} %'.format(violated,percViol))

    return [ValDice1,targetSizeArrBatches_val, predSizeArrBatches_val]

def analyzeViolationContraints(sizeGT,sizePred, minVal, maxVal):
    
    violPos = 0
    violNeg = 0
    diffPos = []
    diffNeg = []

    # To avoid having the vector empty
    diffNeg.append(0)
    diffPos.append(0)
    
    for i_i in range(len(sizeGT)):
        if sizeGT[i_i] == 0:
            if sizePred[i_i]>0:
                violNeg+=1
                diffNeg.append(sizePred[i_i])
        else:
            if sizePred[i_i]>maxVal:
                violPos+=1
                diffPos.append(sizePred[i_i]-maxVal)
                
            if sizePred[i_i]<minVal:
                violPos+=1
                diffPos.append(minVal-sizePred[i_i])    
    
    violPerc_Neg = violNeg*100/len(sizeGT)
    violPerc_Pos = violPos*100/len(sizeGT)


    if len(diffNeg)>1:
        diffNeg = diffNeg[1:len(diffNeg)]

    if len(diffPos)>1:
        diffPos = diffPos[1:len(diffPos)]
        
    return [ violPerc_Neg,
             violPerc_Pos,
             np.mean(diffNeg),
             np.min(diffNeg),
             np.max(diffNeg),
             np.mean(diffPos),
             np.min(diffPos),
             np.max(diffPos)]
 
     
def inference_multiTask(net, img_batch, batch_size, epoch, deepSupervision):
    total = len(img_batch)

    Dice1 = torch.zeros(total, 2)
    Dice2 = torch.zeros(total, 2)
    Dice3 = torch.zeros(total, 2)
    
    net.eval()

    img_names_ALL = []

    dice = computeDiceOneHot().cuda()
    voldiff = []
    xDiff = []
    yDiff = []

    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)
        Segmentation = to_var(labels)

        
        if deepSupervision == False:
            segmentation_prediction, reg_output = net(MRI)
        else:
            segmentation_prediction, seg3,seg2,seg1 = net(MRI)

        Segmentation_planes = getOneHotSegmentation(Segmentation)
        
        # Regression
        feats = getValuesRegression(labels)
        feats_t = torch.from_numpy(feats).float()
        featsVar = to_var(feats_t)
        
        diff =  reg_output - featsVar 
        diff_np = diff.cpu().data.numpy()
        
        voldiff.append(diff_np[0][0])
        xDiff.append(diff_np[0][1])
        yDiff.append(diff_np[0][2])

                    
        DicesN, Dices1, Dices2, Dices3= dice(segmentation_prediction, Segmentation_planes)

        Dice1[i] = Dices1.data
        Dice2[i] = Dices2.data
        Dice3[i] = Dices3.data
       

        # Save images
        # directory = 'resultBladder'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # filenameImg = os.path.join(directory, "original_image_{}_{}.png".format(epoch, i))
        # filenameMask = os.path.join(directory, "groundTruth_image_{}_{}.png".format(epoch, i))
        # filenamePred = os.path.join(directory, "Prediction_{}_{}.png".format(epoch, i))
    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    ValDice1 = DicesToDice(Dice1)
    ValDice2 = DicesToDice(Dice2)
    ValDice3 = DicesToDice(Dice3)
   
    return [ValDice1,ValDice2,ValDice3, voldiff, xDiff, yDiff]
    
def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()


def resizeTensorMask(batch, scalingFactor):
    data = batch.cpu().data.numpy()
    batch_s = data.shape[0]
    numClasses = data.shape[1]
    img_size = data.shape[2]
    # TODO: Better way to define this
    resizedLabels = np.zeros((batch_s,
                              numClasses,
                              img_size / scalingFactor,
                              img_size / scalingFactor))

    for i in range(data.shape[0]):

        for l in range(numClasses):
            img = data[i, l, :, :].reshape(img_size, img_size)
            imgRes = skiTransf.resize(img, (img_size / scalingFactor, img_size / scalingFactor), preserve_range=True)
            idx0 = np.where(imgRes < 0.5)
            idx1 = np.where(imgRes >= 0.5)
            imgRes[idx0] = 0
            imgRes[idx1] = 1
            resizedLabels[i, l, :, :] = imgRes

    tensorClass = torch.from_numpy(resizedLabels).float()
    return Variable(tensorClass.cuda())


def resizeTensorMaskInSingleImage(batch, scalingFactor):
    data = batch.cpu().data.numpy()
    batch_s = data.shape[0]
    numClasses = data.shape[1]
    img_size = data.shape[2]
    # TODO: Better way to define this
    resizedLabels = np.zeros((batch_s,
                              img_size / scalingFactor,
                              img_size / scalingFactor))

    for i in range(data.shape[0]):
        img = data[i, :, :].reshape(img_size, img_size)
        imgL = np.zeros((img_size, img_size))
        idx1t = np.where(img == 1)
        imgL[idx1t] = 1
        imgRes = skiTransf.resize(imgL, (img_size / scalingFactor, img_size / scalingFactor), preserve_range=True)
        idx1 = np.where(imgRes >= 0.5)

        imgL = np.zeros((img_size, img_size))
        idx2t = np.where(img == 1)
        imgL[idx2t] = 1
        imgRes = skiTransf.resize(imgL, (img_size / scalingFactor, img_size / scalingFactor), preserve_range=True)
        idx2 = np.where(imgRes >= 0.5)

        imgL = np.zeros((img_size, img_size))
        idx3t = np.where(img == 1)
        imgL[idx3t] = 1
        imgRes = skiTransf.resize(imgL, (img_size / scalingFactor, img_size / scalingFactor), preserve_range=True)
        idx3 = np.where(imgRes >= 0.5)

        imgResized = np.zeros((img_size / scalingFactor, img_size / scalingFactor))
        imgResized[idx1] = 1
        imgResized[idx2] = 2
        imgResized[idx3] = 3

        resizedLabels[i, :, :] = imgResized

    tensorClass = torch.from_numpy(resizedLabels).long()
    return Variable(tensorClass.cuda())


# TODO : use lr_scheduler from torch.optim
def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


# TODO : use lr_scheduler from torch.optim
def adjust_learning_rate(lr_args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_args * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print(" --- Learning rate:  {}".format(lr))


if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms

    loader = transforms.Compose([transforms.ToTensor()])
    pred = to_var(getOneHotSegmentation(loader(Image.open('MICCAI_Bladder/val/GT/newR12_Lab_65.png')).unsqueeze(0)))
    GT = to_var(getOneHotSegmentation(loader(Image.open('MICCAI_Bladder/val/GT/newR12_Lab_93.png')).unsqueeze(0)))
    print(pred)

    hausdorff = Hausdorff().cuda()

    from time import time

    tic = time()
    for _ in range(50):
        x = hausdorff(pred, GT)
    toc = time()
    print(x, (toc - tic) / 50)
