from utils import *

    
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
        eps = 1e-10
        
        oneHotLabels = torch.cat((weakLabels == 0, weakLabels == 1), dim=0).view(1,numClasses,target.shape[2],target.shape[3]).float()

        numPixelsNonMasked = weakLabels.sum()
        
        #m = input.shape[2]*input.shape[3]
        
        softmax_y = input.cpu().data.numpy()

        # Mask the predictions to only annotated pixels
        mask=oneHotLabels
        mask[:,0,:,:]=0
        
        if (numPixelsNonMasked.cpu().data.numpy() > 0)[0]:
            # This would be if using the logits as input
            grad_input =  ((torch.Tensor(softmax_y).cuda() - torch.Tensor(oneHotLabels.cpu().data).cuda()))*(torch.Tensor(mask.cpu().data).cuda())/(torch.Tensor(numPixelsNonMasked.cpu().data.numpy()).cuda())  # Divide by m or numPixelsNonMasked?
            #grad_input =  ((torch.Tensor(softmax_y).cuda() - torch.Tensor(oneHotLabels.cpu().data).cuda()))*(torch.Tensor(mask.cpu().data).cuda()) # Divide by m or numPixelsNonMasked?
            # If we use the softmax as input (aCE/aSoftmax)
            
            #grad_input = -(torch.Tensor(oneHotLabels.cpu().data).cuda()/torch.Tensor(softmax_y+eps).cuda())*(torch.Tensor(mask.cpu().data).cuda())*(1/100)
            #grad_input = -(torch.Tensor(oneHotLabels.cpu().data).cuda()/torch.Tensor(softmax_y+eps).cuda())*(torch.Tensor(mask.cpu().data).cuda())*(1/torch.Tensor(numPixelsNonMasked.cpu().data.numpy()).cuda())
            #grad_input = -(torch.Tensor(oneHotLabels.cpu().data).cuda()/torch.Tensor(softmax_y+eps).cuda())*(torch.Tensor(mask.cpu().data).cuda())
            
            #np.unique(grad_input.cpu().numpy())
            
            #grad_input =  ((torch.Tensor(softmax_y).cuda() - torch.Tensor(oneHotLabels.cpu().data).cuda()))*(torch.Tensor(mask.cpu().data).cuda())/(torch.Tensor(numPixelsNonMasked.cpu().data.numpy()).cuda())  # Divide by m or numPixelsNonMasked?
        else:
            #grad_input = 0.0
            grad_input =  torch.FloatTensor(1)
            grad_input.fill_(0.0)
            
        #grad_input = grad_input/100
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
        
        
