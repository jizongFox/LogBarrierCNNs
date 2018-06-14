from utils import *


#### High-order size loss with log barrier approximation #######
####      ----------    ONE BOUND --------------  #######

class mySize_Loss_LOG_BARRIER_ONE_BOUND(torch.autograd.Function):

    def forward(self, input, target, upper_B, t ):
        eps = 1e-10
        self.save_for_backward(input, target, upper_B, t)
        
        m = input.shape[2]*input.shape[3]
        
        # Compute the hard size of the prediction
        softmax_y = input.cpu().numpy()
        softB = softmax_y[:,1,:,:]

        # Soft Dice
        sizePredNumpy = softB.sum()
        sizePred = torch.FloatTensor(1)
        sizePred.fill_(sizePredNumpy.item())# This is to try to fix a type issue I got:

        # Let's use the target (annotation) to know whether there some exist some target or not
        
        y = sizePred - upper_B
        y_numpy = y.numpy()
        t_numpy = t.numpy()

        pdb.set_trace()
        if (target[:,1,:,:].sum() > 0 ):
            if (y_numpy <= - (1/(t_numpy**2))):
                loss = - (1/t)*np.log(-y)
            else:
                loss = t*y - (1/t)*np.log(1/(t**2)) + 1/t
        else:
            loss = ((sizePred)**2)/(softB.shape[1]*softB.shape[2])
            #lossVal = ((sizePred)**2)/(softB.shape[1]*softB.shape[2])
            #loss =  torch.FloatTensor(1)
            #loss.fill_(lossVal)
        
        # TO-DO. Currently, the loss is weighted by a hard-coded value (100). Add this as input parameter
        #lossT =  torch.FloatTensor(1)
        #lossT.fill_(loss.numpy()[0]/100)
        lossT = loss/(100*m)
        
        if (np.isnan(loss.numpy()[0])):
            pdb.set_trace()
            
        return lossT.cuda()   # a single number (averaged loss over batch samples)

    def backward(self, grad_output):
        input, target, upper_B, t = self.saved_variables
        numClasses = 2
        eps = 1e-10
        
        numPixelsNonMasked = target[:,1,:,:].sum()
        
        m = input.shape[2]*input.shape[3]
        
        # Compute the soft size of the prediction
        softmax_y = input.cpu().data.numpy()
        softB = softmax_y[:,1,:,:]

        # Soft Dice
        sizePred = softB.sum()
        sizePredNumpy = softB.sum()
        sizePred = torch.FloatTensor(1)
        sizePred.fill_(sizePredNumpy.item())

        y = sizePred - upper_B.data
        y_numpy = y.numpy()
        t_numpy = t.data.numpy()

        # TO-DO. Currently, the loss is weighted by a hard-coded value (100). Add this as input parameter
        if (target[:,1,:,:].sum().cpu().data.numpy()[0] > 0 ):
            if (y_numpy[0] <= - (1/(t_numpy[0]**2))):
                lossValue = (1/(t_numpy*y_numpy))
                lossValue = Variable(torch.Tensor(torch.from_numpy(lossValue)))
                #lossValue = (1/(t_numpy*y_numpy))
            else:
                lossValue = t_numpy
                lossValue = Variable(torch.Tensor(torch.from_numpy(lossValue)))
                #lossValue = t_numpy
        else:
            lossValue = 2 * (sizePred)
            #lossVal = 2 * (sizePredNumpy)/(100*m)
            #lossValue =  torch.FloatTensor(1)
            #lossValue.fill_(lossVal)
            #lossValue = Variable(torch.Tensor(lossValue))
            lossValue = Variable(lossValue)
            
        grad_inputA = np.zeros((softmax_y.shape[0],1,softmax_y.shape[2],softmax_y.shape[3]),dtype='float32')
        grad_inputB = np.zeros((softmax_y.shape[0],1,softmax_y.shape[2],softmax_y.shape[3]),dtype='float32')

        lossValue = lossValue/(100*m)
        grad_inputB.fill(lossValue.data.numpy()[0]) #OPTION A
        #grad_inputB.fill(lossValue[0]) #OPTION B
        #grad_inputB.fill(lossValue.numpy()[0])
        
        grad_input = np.concatenate((grad_inputA,grad_inputB), 1)

        return torch.Tensor(grad_input).cuda(), None, None, None  # Number of returned gradients must be the same as input variables




class mySize_Loss_LOG_BARRIER_TWO_BOUNDS(torch.autograd.Function):

    def forward(self, input, target, lower_B, upper_B, t ):
        eps = 1e-10
        self.save_for_backward(input, target, lower_B, upper_B, t)
        
        m = input.shape[2]*input.shape[3]
        
        # Compute the hard size of the prediction
        softmax_y = input.cpu().numpy()
        softB = softmax_y[:,1,:,:]

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

        
        y_upper = (sizePred - upper_B)
        y_lower = (lower_B - sizePred)
        y_upper_numpy = y_upper.numpy()
        y_lower_numpy = y_lower.numpy()
        t_numpy = t.numpy()

        y_neg_numpy = sizePredNumpy
        y_neg = sizePred
        
        if (target[:,1,:,:].sum() > 0 ):
            # Loss for H1
            if (y_upper_numpy <= - (1/(t_numpy**2))):
                loss_upper = - (1/t)*np.log(-y_upper)
            else:
                loss_upper = t*y_upper - (1/t)*np.log(1/(t**2)) + 1/t

            # Loss for H2
            if (y_lower_numpy <= - (1/(t_numpy**2))):
                loss_lower = - (1/t)*np.log(-y_lower)
            else:
                loss_lower = t*y_lower - (1/t)*np.log(1/(t**2)) + 1/t

            loss = loss_upper + loss_lower
        else:
            #loss = ((sizePred)**2)/(softB.shape[1]*softB.shape[2])
            if (y_neg_numpy <= - (1/(t_numpy**2))):
                loss = - (1/t)*np.log(-y_neg)
            else:
                loss = t*y_neg - (1/t)*np.log(1/(t**2)) + 1/t
            
            #lossVal = ((sizePred)**2)/(softB.shape[1]*softB.shape[2])
            #loss =  torch.FloatTensor(1)
            #loss.fill_(lossVal)
        
        # TO-DO. Currently, the loss is weighted by a hard-coded value (100). Add this as input parameter
        #lossT =  torch.FloatTensor(1)
        #lossT.fill_(loss.numpy()[0]/100)
        lossT = loss/(10*m)
        
        if (np.isnan(loss.numpy()[0])):
            pdb.set_trace()
            
        return lossT.cuda()   # a single number (averaged loss over batch samples)

    def backward(self, grad_output):
        input, target, lower_B, upper_B, t = self.saved_variables
        numClasses = 2
        eps = 1e-10
        
        numPixelsNonMasked = target[:,1,:,:].sum()
        
        m = input.shape[2]*input.shape[3]
        
        # Compute the soft size of the prediction
        softmax_y = input.cpu().data.numpy()
        softMax_pyTorch = input.cpu().data
        softB = softmax_y[:,1,:,:]
        softB_pyTorch = softMax_pyTorch[:,1,:,:]
        
        # Soft Dice
        #sizePred = softB.sum()
        sizePredNumpy = softB.sum()
        sizePred = torch.FloatTensor(1)
        sizePred.fill_(sizePredNumpy.item())
        sizePred = softB_pyTorch.sum()
        
        y_upper = (sizePred - upper_B.data)
        y_lower = (lower_B.data - sizePred)
        y_upper_numpy = y_upper.numpy()
        y_lower_numpy = y_lower.numpy()
        y_neg_numpy = y_lower_numpy
        y_neg_numpy[0] = sizePredNumpy # Workaround to not complain in line 251 : grad_inputB.fill(lossValue.data.numpy()[0]) #OPTION A
        y_neg = sizePred
        t_numpy = t.data.numpy()
        #pdb.set_trace()
        # TO-DO. Currently, the loss is weighted by a hard-coded value (100). Add this as input parameter
        if (target[:,1,:,:].sum().cpu().data.numpy()[0] > 0 ):
            # Loss for H1
            if (y_upper_numpy[0] <= - (1/(t_numpy[0]**2))):
                #lossValue_upper = (1/(t_numpy*y_upper_numpy))

                lossValue_upper = -(1/(t.data*y_upper))
                #lossValue = Variable(torch.Tensor(torch.from_numpy(lossValue)))
                #lossValue = (1/(t_numpy*y_numpy))
            else:
                lossValue_upper = t.data
                #lossValue = Variable(torch.Tensor(torch.from_numpy(lossValue)))
                #lossValue = t_numpy

            # Loss for H2
            if (y_lower_numpy[0] <= - (1/(t_numpy[0]**2))):
                #lossValue_lower = (1/(t_numpy*y_lower_numpy))
                lossValue_lower = -(1/(t.data*y_lower))
                #lossValue = Variable(torch.Tensor(torch.from_numpy(lossValue)))
                #lossValue = (1/(t_numpy*y_numpy))
            else:
                #lossValue_lower = t_numpy
                lossValue_lower = t.data
                #lossValue = Variable(torch.Tensor(torch.from_numpy(lossValue)))

            lossValue = lossValue_lower + lossValue_upper
            #lossValue = Variable(torch.Tensor(torch.from_numpy(lossValue)))
            
        else:
            if (y_neg_numpy[0] <= - (1/(t_numpy[0]**2))):
                #lossValue = (1/(t_numpy*y_neg_numpy))
                lossValue = -(1/(t.data*sizePred))
                #lossValue = Variable(torch.Tensor(torch.from_numpy(lossValue)))
                #lossValue = (1/(t_numpy*y_numpy))
            else:
                #lossValue = t_numpy
                lossValue = t.data

            #lossValue = Variable(torch.Tensor(torch.from_numpy(lossValue)))
            '''lossValue = 2 * (sizePred)
            #lossVal = 2 * (sizePredNumpy)/(100*m)
            #lossValue =  torch.FloatTensor(1)
            #lossValue.fill_(lossVal)
            #lossValue = Variable(torch.Tensor(lossValue))
            lossValue = Variable(lossValue)'''
            
        grad_inputA = np.zeros((softmax_y.shape[0],1,softmax_y.shape[2],softmax_y.shape[3]),dtype='float32')
        grad_inputB = np.zeros((softmax_y.shape[0],1,softmax_y.shape[2],softmax_y.shape[3]),dtype='float32')

        lossValue = lossValue/(100*m)
        #grad_inputB.fill(lossValue.data.numpy()[0]) #OPTION A
        grad_inputB.fill(lossValue[0]) #OPTION B
        #grad_inputB.fill(lossValue.numpy()[0])
        
        grad_input = np.concatenate((grad_inputA,grad_inputB), 1)

        return torch.Tensor(grad_input).cuda(), None, None, None, None  # Number of returned gradients must be the same as input variables
