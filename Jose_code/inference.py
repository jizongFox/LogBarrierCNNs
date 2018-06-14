from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar
from tqdm import tqdm
import medicalDataLoader
from UNet import *
from utils import *
import pdb

from ENet import *

import sys


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def resizeTensorMaskInSingleImage(batch, scalingFactor):
    data = batch.cpu().data.numpy()
    batch_s = data.shape[0]
    numClasses = data.shape[1]
    img_size = data.shape[2]
    # TODO: Better way to define this
    resizedLabels = np.zeros((batch_s,
                              int(img_size/scalingFactor),
                              int(img_size/scalingFactor)))
    
                            
    for i in range(data.shape[0]):
        img = data[i,:,:].reshape(img_size,img_size)
        imgL = np.zeros((img_size,img_size))
        idx1t = np.where(img==1)
        imgL[idx1t]=1
        imgRes = skiTransf.resize(imgL,(img_size/scalingFactor,img_size/scalingFactor),preserve_range=True)
        idx1 = np.where(imgRes>=0.5)
        
        imgL = np.zeros((img_size,img_size))
        idx2t = np.where(img==1)
        imgL[idx2t]=1
        imgRes = skiTransf.resize(imgL,(img_size/scalingFactor,img_size/scalingFactor),preserve_range=True)
        idx2 = np.where(imgRes>=0.5)
        
        imgL = np.zeros((img_size,img_size))
        idx3t = np.where(img==1)
        imgL[idx3t]=1
        imgRes = skiTransf.resize(imgL,(img_size/scalingFactor,img_size/scalingFactor),preserve_range=True)
        idx3 = np.where(imgRes>=0.5)
        
        imgResized = np.zeros((int(img_size/scalingFactor),int(img_size/scalingFactor)))
        imgResized[idx1]=1
        imgResized[idx2]=2
        imgResized[idx3]=3
        
        
        resizedLabels[i,:,:]=imgResized
            
    tensorClass = torch.from_numpy(resizedLabels).long()
    return Variable(tensorClass.cuda())





    
# My CE loss in the case of weakly supervised (some pixels are annotated)  
class myCE_Loss_Weakly_numpy(torch.autograd.Function):

    def forward(self, input, target):
        self.save_for_backward(input, target)
        
        expNumA = np.exp(input[:,0,:,:].cpu().numpy())
        expNumB = np.exp(input[:,1,:,:].cpu().numpy())
        expDen = np.exp(input.cpu().numpy())
        expDen = np.sum(expDen, axis=1)
        softA = expNumA/expDen
        softB = expNumB/expDen
        
        gt_numpy = target.cpu().numpy()

        
        numPixelsNonMasked = target.sum()
        #loss = (- np.sum(np.log(softA)*(1-gt_numpy)) - np.sum(np.log(softB)*(gt_numpy)))/(gt_numpy.shape[1]*gt_numpy.shape[2])
        loss = - np.sum(np.log(softB)*(gt_numpy))/(numPixelsNonMasked+0.0000001)

        lossT =  torch.FloatTensor(1)
        lossT.fill_(loss)
        lossT = lossT.cuda()

        return lossT.cuda()   # a single number (averaged loss over batch samples)

    def backward(self, grad_output):
        input, target = self.saved_variables
        numClasses = 2
        targetOneHot = np.zeros((1,numClasses,target.shape[1],target.shape[2]))
        
        oneHotLabels = torch.cat((target == 0, target == 1), dim=0).view(1,numClasses,target.shape[1],target.shape[2]).float()

        numPixelsNonMasked = target.sum()
        
        m = input.shape[2]*input.shape[3]
        
        # Softmax  (It can be saved with save_for_backward??)
        expNumA = np.exp(input[:,0,:,:].cpu().data.numpy())
        expNumB = np.exp(input[:,1,:,:].cpu().data.numpy())
        expDen = np.exp(input.cpu().data.numpy())
        expDen = np.sum(expDen, axis=1)
        softA = expNumA/expDen
        softB = expNumB/expDen
        softmax_y = np.concatenate((np.reshape(softA,(1,1,softA.shape[1],softA.shape[2])), np.reshape(softB,(1,1,softB.shape[1],softB.shape[2]))), axis=1)

        # Mask the predictions to only annotated pixels
        mask=oneHotLabels
        mask[:,0,:,:]=0
        grad_input =  ((torch.Tensor(torch.Tensor(softmax_y) - torch.Tensor(oneHotLabels.cpu().data))))*torch.Tensor(mask.cpu().data)/m  # Divide by m or numPixelsNonMasked
        
        #pdb.set_trace()
        #pixelsClassA = np.where( softmax_y > 0.5 )
        #sizePred = len(pixelsClassA[0])
        #sizeGT = oneHotLabels[:,1,:,:].sum().cpu().data.numpy()

        #grad_input = np.zeros((softmax_y.shape),dtype='float32')
        #grad_input.fill(2 * (sizePred-sizeGT[0])/m)
        
        #grad_input =  (torch.Tensor(grad_input))
        
        #grad_input = torch.LongTensor(1, 2, 256, 256).fill_(0).float()
        #return None, None
        return grad_input.cuda(), None

def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

    
class myCE_Loss_Weakly_numpy_OneHot(torch.autograd.Function):

    def forward(self, input, target, weakLabels):
        self.save_for_backward(input, target, weakLabels)

        eps = 1e-10
        # Unstable Softmax
        '''expNumA = np.exp(input[:,0,:,:].cpu().numpy())
        expNumB = np.exp(input[:,1,:,:].cpu().numpy())
        expDen = np.exp(input.cpu().numpy())
        expDen = np.sum(expDen, axis=1)
        
        softA = expNumA/expDen
        softB = expNumB/expDen'''

        
        # Stable softmax
        input_numpy = input.cpu().numpy()
        exps = np.exp(input_numpy - np.max(input_numpy))
        sofMax = exps / (np.sum(exps, axis=1))
        
        
        #softMax_pyTorch = nn.Softmax()
        #softMax_pyTorch.cuda()
        #sofM = softMax_pyTorch(input)
        
        #pdb.set_trace()
        '''if( np.isnan(expDen).sum() > 0):
            #print(' Nan is on the denominator...')
            input_numpy = input[:,:,:,:].cpu().numpy()
            nanFinder = np.isnan(expDen)
            idx = np.where(nanFinder==True)
            pdb.set_trace()
            #print('INPUT min: {} and max: {}'.format(np.min(input_numpy), np.max(input_numpy)))

            #print(' Den pix value: {}'.format(expDen[idx]))
            #print(' expNumA pix value: {}'.format(expNumA[idx]))
            #print(' expNumB pix value: {}'.format(expNumB[idx]))'''
        
        '''gt_numpy = (target[:,1,:,:].cpu().numpy()).reshape((256,256))
        gt_eroded = gt_numpy
        
        if (gt_numpy.sum()>0):
            # Erode it
            struct2 = ndimage.generate_binary_structure(2, 3)
            gt_eroded = ndimage.binary_erosion(gt_numpy, structure=struct2,iterations=7).astype(gt_numpy.dtype)
        
            # To be sure that we do not delete the Weakly annoated label
            if (gt_eroded.sum() == 0):
                gt_eroded = ndimage.binary_erosion(gt_numpy, structure=struct2,iterations=3).astype(gt_numpy.dtype)
        
        
        numPixelsNonMasked = gt_eroded.sum()
        #loss = (- np.sum(np.log(softA)*(1-gt_numpy)) - np.sum(np.log(softB)*(gt_numpy)))/(gt_numpy.shape[1]*gt_numpy.shape[2])
        loss = - np.sum(np.log(softB)*(gt_eroded))/(numPixelsNonMasked+0.0000000000001)

        lossT =  torch.FloatTensor(1)
        lossT.fill_(np.float32(loss).item())
        lossT = lossT.cuda()'''

        numPixelsNonMasked = weakLabels.sum()
        #loss = (- np.sum(np.log(softA)*(1-gt_numpy)) - np.sum(np.log(softB)*(gt_numpy)))/(gt_numpy.shape[1]*gt_numpy.shape[2])

        # With the non-stable version
        #loss = - np.sum(np.log(softB)*(weakLabels.view(1,256,256)).cpu().numpy())/(numPixelsNonMasked+0.0000001)

        # With the stable version
        loss = - np.sum(np.log(sofMax[:,1,:,:])*(weakLabels.view(1,256,256)).cpu().numpy())/(numPixelsNonMasked+eps)

        
        lossT =  torch.FloatTensor(1)
        lossT.fill_(np.float32(loss).item())

        if (np.isnan(loss)):
            pdb.set_trace()
            
        lossT = lossT.cuda()
        
        return lossT.cuda()   # a single number (averaged loss over batch samples)

    def backward(self, grad_output):
        input, target,weakLabels = self.saved_variables
        numClasses = 2
        eps = 1e-10

        
        #gt_numpy = (target[:,1,:,:].cpu().numpy()).reshape((256,256))
        '''gt_numpy = (target[:,1,:,:].cpu().data.numpy()).reshape((256,256))
        gt_eroded = gt_numpy
        
        if (gt_numpy.sum()>0):
            # Erode it
            struct2 = ndimage.generate_binary_structure(2, 3)
            gt_eroded = ndimage.binary_erosion(gt_numpy, structure=struct2,iterations=7).astype(gt_numpy.dtype)
        
            # To be sure that we do not delete the Weakly annoated label
            if (gt_eroded.sum() == 0):
                gt_eroded = ndimage.binary_erosion(gt_numpy, structure=struct2,iterations=3).astype(gt_numpy.dtype)
                
         
        gt_eroded_Torch = torch.from_numpy(gt_eroded.reshape((1,256,256))).float()
        gt_eroded_Torch = gt_eroded_Torch.cuda() '''

        oneHotLabels = torch.cat((weakLabels == 0, weakLabels == 1), dim=0).view(1,numClasses,target.shape[2],target.shape[3]).float()

        numPixelsNonMasked = weakLabels.sum()
        
        m = input.shape[2]*input.shape[3]
        
        # Softmax  (It can be saved with save_for_backward??)
        # Unstable Softmax
        '''
        expNumA = np.exp(input[:,0,:,:].cpu().data.numpy())
        expNumB = np.exp(input[:,1,:,:].cpu().data.numpy())
        expDen = np.exp(input.cpu().data.numpy())
        expDen = np.sum(expDen, axis=1)
        softA = expNumA/expDen
        softB = expNumB/expDen
        softmax_y = np.concatenate((np.reshape(softA,(1,1,softA.shape[1],softA.shape[2])), np.reshape(softB,(1,1,softB.shape[1],softB.shape[2]))), axis=1)'''

        # Stable softmax
        input_numpy = input.cpu().data.numpy()
        exps = np.exp(input_numpy - np.max(input_numpy))
        softmax_y = exps / (np.sum(exps, axis=1))

        # Mask the predictions to only annotated pixels
        mask=oneHotLabels
        mask[:,0,:,:]=0
  
        #grad_input =  ((torch.Tensor(torch.Tensor(softmax_y) - torch.Tensor(oneHotLabels.cpu().data))))*torch.Tensor(mask.cpu().data)/m  # Divide by m or numPixelsNonMasked
        #grad_input =  ((torch.Tensor(softmax_y).cuda() - oneHotLabels))*(mask)/(numPixelsNonMasked+0.000000000001)  # Divide by m or numPixelsNonMasked
        grad_input =  ((torch.Tensor(softmax_y).cuda() - torch.Tensor(oneHotLabels.cpu().data).cuda()))*(torch.Tensor(mask.cpu().data).cuda())/(torch.Tensor(numPixelsNonMasked.cpu().data.numpy()+eps).cuda())  # Divide by m or numPixelsNonMasked
        
        return grad_input.cuda(), None, None
 

class myCE_Loss_Weakly_numpy_OneHot_SoftMaxPyTorch(torch.autograd.Function):

    def forward(self, input, target, weakLabels):
        self.save_for_backward(input, target, weakLabels)

        eps = 1e-20
        
        softmax_y = input.cpu().numpy()
        numPixelsNonMasked = weakLabels.sum()
        
        if (numPixelsNonMasked > 0):
        # Mask the non-annotated pixels
            loss = - np.sum(np.log(softmax_y[:,1,:,:]+eps)*(weakLabels.view(1,256,256)).cpu().numpy())/(numPixelsNonMasked)
            #loss = - np.sum(np.log(softmax_y[:,1,:,:])*(weakLabels.view(1,256,256)).cpu().numpy())/(numPixelsNonMasked)
        else:
            loss = 0.0
            
        lossT =  torch.FloatTensor(1)
        lossT.fill_(np.float32(loss).item())

        #if (np.isnan(loss)):
        #    pdb.set_trace()
            
        lossT = lossT.cuda()
        
        return lossT.cuda()   # a single number (averaged loss over batch samples)

    def backward(self, grad_output):
        input, target,weakLabels = self.saved_variables
        numClasses = 2
        #eps = 1e-10
        
        oneHotLabels = torch.cat((weakLabels == 0, weakLabels == 1), dim=0).view(1,numClasses,target.shape[2],target.shape[3]).float()

        numPixelsNonMasked = weakLabels.sum()
        
        #m = input.shape[2]*input.shape[3]
        
        softmax_y = input.cpu().data.numpy()

        # Mask the predictions to only annotated pixels
        mask=oneHotLabels
        mask[:,0,:,:]=0
        
        #pdb.set_trace()
        if (numPixelsNonMasked.cpu().data.numpy() > 0)[0]:
            grad_input =  ((torch.Tensor(softmax_y).cuda() - torch.Tensor(oneHotLabels.cpu().data).cuda()))*(torch.Tensor(mask.cpu().data).cuda())/(torch.Tensor(numPixelsNonMasked.cpu().data.numpy()).cuda())  # Divide by m or numPixelsNonMasked?
        else:
            #grad_input = 0.0
            grad_input =  torch.FloatTensor(1, 2, 256, 256)
            #grad_input =  torch.FloatTensor(1)
            grad_input.fill_(0.0)
            
            
        return grad_input.cuda(), None, None

class mySize_Loss_numpy_SoftMaxPyTorch(torch.autograd.Function):

    def forward(self, input, target, lower_B, upper_B):
        eps = 1e-10
        self.save_for_backward(input, target, lower_B, upper_B)
        
        # Compute the hard size of the prediction
        softmax_y = input.cpu().numpy()
        softB = softmax_y[:,1,:,:]

        # Hard-Size
        #pixelsClassB = np.where( softB > 0.5 )
        #sizePredNumpy = len(pixelsClassB[0])/1.0
        #sizePred = torch.FloatTensor(1)
        #sizePred.fill_(sizePredNumpy)
        # Soft Dice
        sizePredNumpy = softB.sum()
        sizePred = torch.FloatTensor(1)
        sizePred.fill_(sizePredNumpy.item())# This is to try to fix a type issue I got:
        
        #TypeError: sub received an invalid combination of arguments - got (numpy.float32), but expected one of:
        #* (float value)
        #    didn't match because some of the arguments have invalid types: (numpy.float32)
        #* (torch.FloatTensor other)
        #    didn't match because some of the arguments have invalid types: (numpy.float32)
        #* (float value, torch.FloatTensor other)
 
        # Let's use the target (annotation) to know whether there some exist some target or not
        #pdb.set_trace()
        if (target[:,1,:,:].sum() > 0 ):
            if (sizePred.numpy()[0] > upper_B.numpy()[0]):
                loss = ((sizePred - upper_B)**2)/(softB.shape[1]*softB.shape[2])
            
            elif (sizePred.numpy()[0] < lower_B.numpy()[0]):
                #loss = ((lower_B - sizePred)**2)/(softB.shape[1]*softB.shape[2])
                loss = ((sizePred - lower_B)**2)/(softB.shape[1]*softB.shape[2])  # Lena's TRUST
            
            else:
                loss =  torch.FloatTensor(1)
                loss.fill_(0)
        else:
            loss = ((sizePred)**2)/(softB.shape[1]*softB.shape[2])
            
            #lossVal = ((sizePred)**2)/(softB.shape[1]*softB.shape[2])
            #loss =  torch.FloatTensor(1)
            
            #loss.fill_(lossVal)
        
        # TO-DO. Currently, the loss is weighted by a hard-coded value (100). Add this as input parameter
        #lossT =  torch.FloatTensor(1)
        #lossT.fill_(loss.numpy()[0]/100)
        lossT = loss/100
        
        if (np.isnan(loss.numpy()[0])):
            pdb.set_trace()
            
        return lossT.cuda()   # a single number (averaged loss over batch samples)

    def backward(self, grad_output):
        input, target, lower_B, upper_B = self.saved_variables
        numClasses = 2
        eps = 1e-10
        
        numPixelsNonMasked = target[:,1,:,:].sum()
        
        m = input.shape[2]*input.shape[3]
        
        # Compute the hard size of the prediction
        softmax_y = input.cpu().data.numpy()
        softB = softmax_y[:,1,:,:]

        # Hard-Size
        #pixelsClassB = np.where( softB > 0.5 )
        #sizePredNumpy = len(pixelsClassB[0])/1.0
        #sizePred = torch.FloatTensor(1)
        #sizePred.fill_(sizePredNumpy)
        
        # Soft Dice
        sizePred = softB.sum()
        sizePredNumpy = softB.sum()
        sizePred = torch.FloatTensor(1)
        sizePred.fill_(sizePredNumpy.item())
        
        # TO-DO. Currently, the loss is weighted by a hard-coded value (100). Add this as input parameter
        if (target[:,1,:,:].sum().cpu().data.numpy()[0] > 0 ):
            if (sizePred.numpy()[0] > upper_B.data.numpy()[0]):
                #lossValue = 2 * (sizePred-upper_B.data.numpy()[0])/(100*m)  OPTION A
                lossValue = 2 * (sizePred-upper_B.data)/(100*m)
                
            elif (sizePred.numpy()[0] < lower_B.data.numpy()[0]):
                #lossValue = 2 * (lower_B.data.numpy()[0] - sizePred)/(100*m)  OPTION A
                #lossValue = 2 * (lower_B.data- sizePred)/(100*m)
                lossValue =  2 * (sizePred-lower_B.data)/(100*m) # Lena's TRUST
            
            else:
                #lossValue =  0.0  OPTION A
                lossValue =  torch.FloatTensor(1)
                lossValue.fill_(0.0)
        else:
            lossValue = 2 * (sizePred)/(100*m)  
            
            
        grad_inputA = np.zeros((softmax_y.shape[0],1,softmax_y.shape[2],softmax_y.shape[3]),dtype='float32')
        grad_inputB = np.zeros((softmax_y.shape[0],1,softmax_y.shape[2],softmax_y.shape[3]),dtype='float32')
        
        #grad_inputB.fill(lossValue) OPTION A
        grad_inputB.fill(lossValue.numpy()[0])
        
        grad_input = np.concatenate((grad_inputA,grad_inputB), 1)

        return torch.Tensor(grad_input).cuda(), None, None, None  # Number of returned gradients must be the same as input variables
        
                        
# My CE loss in the case of weakly supervised (some pixels are annotated)  
class mySize_Loss_numpy(torch.autograd.Function):

    def forward(self, input, target, lower_B, upper_B):
        eps = 1e-10
        self.save_for_backward(input, target, lower_B, upper_B)
        
        expNumA = np.exp(input[:,0,:,:].cpu().numpy())
        expNumB = np.exp(input[:,1,:,:].cpu().numpy())
        expDen = np.exp(input.cpu().numpy())
        expDen = np.sum(expDen, axis=1)
        softA = expNumA/(expDen+eps)
        softB = expNumB/(expDen+eps)

        pixelsClassB = np.where( softB > 0.5 )

        sizePred = len(pixelsClassB[0])
        
        # Let's use the target (annotation) to know whether there some exist some target or not
      
        if (target[:,1,:,:].sum() > 0 ):
            if (sizePred > upper_B.numpy()[0]):
                loss = ((sizePred - upper_B)**2)/(softB.shape[1]*softB.shape[2])
            
            elif (sizePred < lower_B.numpy()[0]):
                loss = ((lower_B - sizePred)**2)/(softB.shape[1]*softB.shape[2])
            
            else:
                loss =  torch.FloatTensor(1)
                loss.fill_(0)
        else:
            lossVal = ((sizePred)**2)/(softB.shape[1]*softB.shape[2])
            loss =  torch.FloatTensor(1)
            loss.fill_(lossVal)
        # Compute the size of the target
        
        lossT =  torch.FloatTensor(1)
        lossT.fill_(loss.numpy()[0]/100)

        if (np.isnan(loss.numpy()[0])):
            pdb.set_trace()
            
        return lossT.cuda()   # a single number (averaged loss over batch samples)

    def backward(self, grad_output):
        input, target, lower_B, upper_B = self.saved_variables
        numClasses = 2
        eps = 1e-10
        
        numPixelsNonMasked = target[:,1,:,:].sum()
        
        m = input.shape[2]*input.shape[3]
        
        # Softmax  (It can be saved with save_for_backward??)
        expNumA = np.exp(input[:,0,:,:].cpu().data.numpy())
        expNumB = np.exp(input[:,1,:,:].cpu().data.numpy())
        expDen = np.exp(input.cpu().data.numpy())
        expDen = np.sum(expDen, axis=1)
        softA = expNumA/(expDen+eps)
        softB = expNumB/(expDen+eps)
        softmax_y = np.concatenate((np.reshape(softA,(1,1,softA.shape[1],softA.shape[2])), np.reshape(softB,(1,1,softB.shape[1],softB.shape[2]))), axis=1)

        
        pixelsClassB = np.where( softB > 0.5 )
        sizePred = len(pixelsClassB[0])
        
        if (target[:,1,:,:].sum().cpu().data.numpy()[0] > 0 ):
            if (sizePred > upper_B.data.numpy()[0]):
                lossValue = 2 * (sizePred-upper_B.data.numpy()[0])/(100*m)
                
            elif (sizePred < lower_B.data.numpy()[0]):
                lossValue = 2 * (lower_B.data.numpy()[0] - sizePred)/(100*m)
            
            else:
                lossValue =  0.0
        else:
            lossValue = 2 * (sizePred - 0)/(100*m)
            
            
        grad_inputA = np.zeros((softmax_y.shape[0],1,softmax_y.shape[2],softmax_y.shape[3]),dtype='float32')
        grad_inputB = np.zeros((softmax_y.shape[0],1,softmax_y.shape[2],softmax_y.shape[3]),dtype='float32')
        
        #grad_inputB.fill(lossValue.data.numpy()[0])
        grad_inputB.fill(lossValue)
        grad_input = np.concatenate((grad_inputA,grad_inputB), 1)
        
        #print( ' Size diff: {} and grad value: {}'.format(sizePred-sizeGT, 2 * (sizePred-sizeGT)/(1000*m)))
        # Mask the predictions to only annotated pixels
   

        return torch.Tensor(grad_input).cuda(), None, None, None  # Number of returned gradients must be the same as input variables
        
        
        
def getOneHot_Encoded_Segmentation(batch):
    backgroundVal = 0
    foregroundVal = 1.0
   
    oneHotLabels = torch.cat((batch == backgroundVal, batch == foregroundVal), dim=1)
    return oneHotLabels.float()
    
            
def runInference(argv):
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    # Batch size for training MUST be 1 in weakly/semi supervised learning if we want to impose constraints.
    batch_size = 1
    batch_size_val = 1
    batch_size_val_save = 1
    batch_size_val_savePng = 1
    lr = 0.0005
    epoch = 1000
 
    root_dir = './ACDC-2D-All'
    model_dir = 'model'


    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    val_set = medicalDataLoader.MedicalImageDataset('val',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            num_workers=5,
                            shuffle=False)
    
                                                    
    val_loader_save_images = DataLoader(val_set,
                                        batch_size=batch_size_val_save,
                                        num_workers=5,
                                        shuffle=False)

    val_loader_save_imagesPng = DataLoader(val_set,
                                        batch_size=batch_size_val_savePng,
                                        num_workers=5,
                                        shuffle=False)
    

    
    # Initialize
    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")
    num_classes = 2

    # ENet
    netG = ENet(1,num_classes)
    #netG = UNetG_Dilated(1,16,4)

    netG.apply(weights_init)
    softMax = nn.Softmax()
    Dice_loss = computeDiceOneHotBinary()
    
 
    
    #CE_loss_weakly_numpy_OneHot = myCE_Loss_Weakly_numpy_OneHot()
    CE_loss_weakly_numpy_OneHot = myCE_Loss_Weakly_numpy_OneHot_SoftMaxPyTorch()
    #sizeLoss = mySize_Loss_numpy()
    sizeLoss = mySize_Loss_numpy_SoftMaxPyTorch()
    
    if torch.cuda.is_available():
        netG.cuda()
        softMax.cuda()
        Dice_loss.cuda()
    
    '''modelName = 'WeaklySupervised_ENet_HardSizeLoss_NewValues'''
    modelName = argv[0]
    try:
        netG = torch.load('./model/Best_' + modelName+'.pkl')
        print("--------model restored--------")
    except:
        print("--------model not restored--------")
        pass
        
    #netG.cuda()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG, mode='max', patience=4, verbose=True,
                                                       factor=10 ** -0.5)
    #optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    BestDice, BestEpoch = 0, 0

    dBAll = []
    percV = []
    dWAll = []
    dTAll = []
    Losses = []
    Losses1 = []

    annotatedPixels = 0
    totalPixels = 0
    
    deepSupervision = False
    saveImagesSegmentation(netG, val_loader_save_imagesPng, batch_size_val_savePng, 0, modelName, deepSupervision)
    
    pdb.set_trace()
    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    for i in range(epoch):
        netG.train()
        lossVal = []
        lossVal1 = []

        totalImages = len(train_loader)
        #d1, percViol = inference(netG, 1.0, val_loader, batch_size, i, False)
        #pdb.set_trace()
        for j, data in enumerate(train_loader):
            image, labels, weak_labels, img_names = data

            # prevent batchnorm error for batch of size 1
            if image.size(0) != batch_size:
                continue

            optimizerG.zero_grad()
            MRI = to_var(image)
            Segmentation = to_var(labels)
            weakAnnotations = to_var(weak_labels)
            target_dice = to_var(torch.ones(1))

            netG.zero_grad()
          
            segmentation_prediction = netG(MRI)
            
            annotatedPixels = annotatedPixels + weak_labels.sum()
            totalPixels = totalPixels + weak_labels.shape[2]*weak_labels.shape[3]
            temperature = 0.1
            predClass_y = softMax(segmentation_prediction/temperature)
            Segmentation_planes = getOneHot_Encoded_Segmentation(Segmentation)
            segmentation_prediction_ones = predToSegmentation(predClass_y)
            #segmentation_prediction_ones = predToSegmentation(segmentation_prediction)
            
            # It needs the logits, not the softmax
            Segmentation_class = getTargetSegmentation(Segmentation)
            
            #lossCE_numpy = CE_loss_weakly_numpy_OneHot(segmentation_prediction, Segmentation_planes, weakAnnotations)
            lossCE_numpy = CE_loss_weakly_numpy_OneHot(predClass_y, Segmentation_planes, weakAnnotations)

            minSize = torch.FloatTensor(1)
            minSize.fill_(np.int64(minVal).item())
            maxSize = torch.FloatTensor(1)
            maxSize.fill_(np.int64(maxVal).item())
            
            #sizeLoss_val = sizeLoss(segmentation_prediction, Segmentation_planes, Variable(minSize), Variable( maxSize))
            #sizeLoss_val = sizeLoss(predClass_y, Segmentation_planes, Variable(minSize), Variable( maxSize))
            
            #CE_lossG = CE_loss(segmentation_prediction, Segmentation_class)
        
            # Dice loss (ONLY USED TO COMPUTE THE DICE. This DICE loss version does not work)
            DicesN, DicesB = Dice_loss(segmentation_prediction_ones, Segmentation_planes)
            DiceN = DicesToDice(DicesN)
            DiceB = DicesToDice(DicesB)
            
            Dice_score = (DiceB + DiceN ) / 2
           
           
            #lossG = lossCE_numpy + sizeLoss_val 
            lossG = lossCE_numpy 

   
            lossG.backward(retain_graph=True)
            #lossG.backward()
            optimizerG.step()
            
            lossVal.append(lossG.data[0])
            lossVal1.append(lossCE_numpy.data[0])


            printProgressBar(j + 1, totalImages,
                             prefix="[Training] Epoch: {} ".format(i),
                             length=15,
                             suffix=" Mean Dice: {:.4f}, Dice1: {:.4f} ".format(
                                 Dice_score.data[0],
                                 DiceB.data[0]))

        deepSupervision = False
        if deepSupervision == False:
            '''printProgressBar(totalImages, totalImages,
                             done="[Training] Epoch: {}, LossG: {:.4f},".format(i,np.mean(lossVal),np.mean(lossVal1)))'''
            printProgressBar(totalImages, totalImages,
                             done="[Training] Epoch: {}, LossG: {:.4f}, lossMSE: {:.4f}".format(i,np.mean(lossVal),np.mean(lossVal1)))
        else:
            printProgressBar(totalImages, totalImages,
                             done="[Training] Epoch: {}, LossG: {:.4f}, Loss4: {:.4f}, Loss3: {:.4f}, Loss2: {:.4f}, Loss1: {:.4f}".format(i,
                                                                                                                                           np.mean(lossVal),
                                                                                                                                           np.mean(lossVal1),
                                                                                                                                           np.mean(lossVal05),
                                                                                                                                           np.mean(lossVal025),
                                                                                                                                           np.mean(lossVal0125)))

        # Save statistics
        modelName = 'WeaklySupervised_ENet_SoftSizeLoss_asTRUST_Temp01_OneBound_50000'
        #modelName = 'Test'
        Losses.append(np.mean(lossVal))

        d1, percViol = inference(netG, temperature, val_loader, batch_size, i, deepSupervision)

             
        '''if currentDice > BestDice:
            BestDice = currentDice
            BestDiceT = d1
            BestEpoch = i
            if np.mean(currentDice) > 0.88:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Saving best model..... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                #torch.save(netG, os.path.join(model_dir, "Best_" + modelName + ".pkl"))

                # Save images
                #saveImages(netG, val_loader_save_imagesPng, batch_size_val_savePng, i, deepSupervision)
                #saveImagesAsMatlab(netG, val_loader_save_images, batch_size_val_save, i)
                #saveImagesAsMatlab(netG, val_loader_save_images_york, batch_size_val_save, i)
        print("###                                                       ###")
        print("###    Best Dice: {:.4f} at epoch {} with DiceT: {:.4f}    ###".format(BestDice, BestEpoch, BestDiceT))
        print("###                                                       ###")'''

        if i % (BestEpoch + 10):
            for param_group in optimizerG.param_groups:
                param_group['lr'] = lr
        
        #scheduler.step(currentDice)

if __name__ == '__main__':
    runInference(sys.argv[1:])
