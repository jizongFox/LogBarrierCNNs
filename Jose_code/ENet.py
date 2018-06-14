from Blocks import *
import torch.nn.init as init
import torch.nn.functional as F
import pdb
import math
from layers import *
from torch.autograd import Variable
       
       
class classificationBlock(nn.Module):
    def __init__(self,in_dim, hiddenUnits, out_dim):
        super(classificationBlock,self).__init__()
        self.lin0  = nn.Linear(in_dim, hiddenUnits)
        self.ReLU0 = nn.ReLU()
        self.lin1  = nn.Linear(hiddenUnits, int(hiddenUnits/8))
        self.ReLU1 = nn.ReLU()
        self.lin2  = nn.Linear(int(hiddenUnits/8),out_dim)
        
    def forward(self, input):
        lin0  = self.lin0(input)
        relu0 = self.ReLU0(lin0)
        lin1  = self.lin1(relu0)
        relu1 = self.ReLU1(lin1)

        return self.lin2(relu1)
    
         
class BottleNeckDownSampling(nn.Module):

    def __init__(self,in_dim, projectionFactor, out_dim):
        super(BottleNeckDownSampling,self).__init__()
        # Main branch
        self.maxpool0 = nn.MaxPool2d(2,return_indices=True)
        #Secondary branch
        self.conv0 = nn.Conv2d(in_dim,int(in_dim/projectionFactor), kernel_size=2, stride=2)
        self.bn0 = nn.BatchNorm2d(int(in_dim/projectionFactor))
        self.PReLU0 = nn.PReLU()

        self.conv1 = nn.Conv2d(int(in_dim/projectionFactor),int(in_dim/projectionFactor), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(int(in_dim/projectionFactor))
        self.PReLU1 = nn.PReLU()

        '''self.conv2 = nn.Conv2d(int(in_dim/projectionFactor),in_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_dim)
        self.PReLU2 = nn.PReLU()'''

        self.block2 = conv_block_1(int(in_dim/projectionFactor),out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        # Main branch
        maxpool_output, indices = self.maxpool0(input)
        
        # Secondary branch
        c0 = self.conv0(input)
        b0 = self.bn0(c0)
        p0 = self.PReLU0(b0)
    
        c1 = self.conv1(p0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)
        
        '''c2 = self.conv2(p1)
        b2 = self.bn2(c2)
        p2 = self.PReLU2(b2)'''

        p2 = self.block2(p1)
        
        do = self.do(p2)

        # Zero padding the feature maps from the main branch
        depth_to_pad = abs(maxpool_output.shape[1] - do.shape[1])
        padding = Variable(torch.zeros(maxpool_output.shape[0], depth_to_pad, maxpool_output.shape[2], maxpool_output.shape[3]).cuda())
        maxpool_output_pad = torch.cat((maxpool_output, padding), 1)
        output = maxpool_output_pad + do
        output = self.PReLU3(output)
        
        return output, indices

        
class BottleNeckDownSamplingDilatedConv(nn.Module):

    def __init__(self,in_dim, projectionFactor, out_dim, dilation):
        super(BottleNeckDownSamplingDilatedConv,self).__init__()
        # Main branch
        
        #Secondary branch
        self.block0 = conv_block_1(in_dim,int(in_dim/projectionFactor))

        self.conv1 = nn.Conv2d(int(in_dim/projectionFactor),int(in_dim/projectionFactor), kernel_size=3, padding=dilation, dilation =dilation)
        self.bn1 = nn.BatchNorm2d(int(in_dim/projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim/projectionFactor),out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        
        # Secondary branch
        b0 = self.block0(input)
 
        c1 = self.conv1(b0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)
        
        b2 = self.block2(p1)
        
        do = self.do(b2)

        output = input + do
        output = self.PReLU3(output)
        
        return output

class BottleNeckDownSamplingDilatedConvLast(nn.Module):

    def __init__(self,in_dim, projectionFactor, out_dim, dilation):
        super(BottleNeckDownSamplingDilatedConvLast,self).__init__()
        # Main branch
        
        #Secondary branch
        self.block0 = conv_block_1(in_dim,int(in_dim/projectionFactor))

        self.conv1 = nn.Conv2d(int(in_dim/projectionFactor),int(in_dim/projectionFactor), kernel_size=3, padding=dilation, dilation =dilation)
        self.bn1 = nn.BatchNorm2d(int(in_dim/projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim/projectionFactor),out_dim)

        self.do = nn.Dropout(p=0.01)
        self.conv_out = nn.Conv2d(in_dim,out_dim, kernel_size=3, padding=1)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        
        # Secondary branch
        b0 = self.block0(input)
 
        c1 = self.conv1(b0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)
        
        b2 = self.block2(p1)
        
        do = self.do(b2)

        output = self.conv_out(input) + do
        output = self.PReLU3(output)
        
        return output
        
class BottleNeckNormal(nn.Module):

    def __init__(self,in_dim, out_dim, projectionFactor, dropoutRate):
        super(BottleNeckNormal,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Main branch

        #Secondary branch
        self.block0 = conv_block_1(in_dim,int(in_dim/projectionFactor))
        self.block1 = conv_block_3_3(int(in_dim/projectionFactor),int(in_dim/projectionFactor))
        self.block2 = conv_block_1(int(in_dim/projectionFactor),out_dim)

        self.do = nn.Dropout(p=dropoutRate)
        self.PReLU_out = nn.PReLU()
        
        if in_dim > out_dim:
            self.conv_out = conv_block_1(in_dim,out_dim)
            
    def forward(self, input):
        # Main branch
        # Secondary branch
        b0 = self.block0(input)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        if self.in_dim > self.out_dim:
            output = self.conv_out(input) + do
        else: 
            output = input + do
        output = self.PReLU_out(output)

        return output
        

class BottleNeckNormal_Asym(nn.Module):

    def __init__(self,in_dim, out_dim, projectionFactor, dropoutRate):
        super(BottleNeckNormal_Asym,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Main branch

        #Secondary branch
        self.block0 = conv_block_1(in_dim,int(in_dim/projectionFactor))
        self.block1 = conv_block_Asym(int(in_dim/projectionFactor),int(in_dim/projectionFactor), 5)
        self.block2 = conv_block_1(int(in_dim/projectionFactor),out_dim)

        self.do = nn.Dropout(p=dropoutRate)
        self.PReLU_out = nn.PReLU()
        
        if in_dim > out_dim:
            self.conv_out = conv_block_1(in_dim,out_dim)
            
    def forward(self, input):
        # Main branch
        # Secondary branch
        b0 = self.block0(input)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        if self.in_dim > self.out_dim:
            output = self.conv_out(input) + do
        else: 
            output = input + do
        output = self.PReLU_out(output)

        return output
                        
class BottleNeckUpSampling(nn.Module):

    def __init__(self,in_dim, projectionFactor, out_dim):
        super(BottleNeckUpSampling,self).__init__()
        # Main branch
        #self.maxunpool0 = nn.MaxPool2d(2)
        #Secondary branch
        self.conv0 = nn.Conv2d(in_dim,int(in_dim/projectionFactor), kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(int(in_dim/projectionFactor))
        self.PReLU0 = nn.PReLU()

        self.conv1 = nn.Conv2d(int(in_dim/projectionFactor),int(in_dim/projectionFactor), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(int(in_dim/projectionFactor))
        self.PReLU1 = nn.PReLU()

        '''self.conv2 = nn.Conv2d(int(in_dim/projectionFactor),in_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_dim)
        self.PReLU2 = nn.PReLU()'''

        self.block2 = conv_block_1(int(in_dim/projectionFactor),out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        # Main branch
        #maxpool_output = self.maxunpool0(input,indices)
        #pdb.set_trace()
        # Secondary branch
        c0 = self.conv0(input)
        b0 = self.bn0(c0)
        p0 = self.PReLU0(b0)
    
        c1 = self.conv1(p0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)
        
        '''c2 = self.conv2(p1)
        b2 = self.bn2(c2)
        p2 = self.PReLU2(b2)'''

        p2 = self.block2(p1)
        
        
        do = self.do(p2)

        #output = maxpool_output + do
        #output = self.PReLU3(output)
        
        return do

                
        
class Conv_residual_conv(nn.Module):

    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv_2 = conv_block_3(self.out_dim, self.out_dim, act_fn)
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3


class unetUp(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_dim, out_dim, act_fn)

        self.up = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)

        return self.conv(torch.cat([outputs1, outputs2], 1))


class unetConv2(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(unetConv2, self).__init__()
        kernel_size = 3
        stride = 1
        padding = 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv1 = conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv2 = conv_block(self.out_dim, self.out_dim, act_fn)

        '''self.conv1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU(),)
                                   
        self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_dim, kernel_size, stride, padding),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU(),)'''

    def forward(self, input):
        conv_1 = self.conv1(input)
        conv_2 = self.conv2(conv_1)

        return conv_2



class UNetG(nn.Module):
    def __init__(self, nin, nG, nout):
        super(UNetG, self).__init__()
        self.conv0 = nn.Sequential(convBatch(nin, nG),
                                   convBatch(nG, nG))
        self.conv1 = nn.Sequential(convBatch(nG * 1, nG * 2, stride=2),
                                   convBatch(nG * 2, nG * 2))
        self.conv2 = nn.Sequential(convBatch(nG * 2, nG * 4, stride=2),
                                   convBatch(nG * 4, nG * 4))
        self.conv3 = nn.Sequential(convBatch(nG * 4, nG * 8, stride=2),
                                   convBatch(nG * 8, nG * 8))
        self.bridge = nn.Sequential(convBatch(nG * 8, nG * 16, stride=2),
                                    residualConv(nG * 16, nG * 16),
                                    convBatch(nG * 16, nG * 16))

        self.deconv0 = upSampleConv(nG * 16, nG * 16)
        self.conv4 = nn.Sequential(convBatch(nG * 24, nG * 8),
                                   convBatch(nG * 8, nG * 8))
        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(convBatch(nG * 12, nG * 4),
                                   convBatch(nG * 4, nG * 4))
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 6, nG * 2),
                                   convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 3, nG * 1),
                                   convBatch(nG * 1, nG * 1))

        self.final = nn.Conv2d(nG, nout, kernel_size=1)

    def forward(self, input):
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        bridge = self.bridge(x3)
        print('  HELLO......')
        y = self.deconv0(bridge)
        y = self.deconv1(self.conv4(torch.cat((y, x3), dim=1)))
        y = self.deconv2(self.conv5(torch.cat((y, x2), dim=1)))
        y = self.deconv3(self.conv6(torch.cat((y, x1), dim=1)))
        y = self.conv7(torch.cat((y, x0), dim=1))

        return F.softmax(self.final(y), dim=1)


class ENet(nn.Module):
    def __init__(self, nin, nout):
        super(ENet, self).__init__()
        self.projectingFactor = 4
        self.numKernelsInit = 16
        # Initial
        self.conv0 = nn.Conv2d(nin,15, kernel_size=3, stride=2, padding=1)
        self.maxpool0 = nn.MaxPool2d(2, return_indices = True)

        # First group
        self.bottleNeck1_0 = BottleNeckDownSampling(self.numKernelsInit,self.projectingFactor, self.numKernelsInit*4)
        self.bottleNeck1_1 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)
        self.bottleNeck1_2 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)
        self.bottleNeck1_3 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)
        self.bottleNeck1_4 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)

        # Second group
        self.bottleNeck2_0 = BottleNeckDownSampling(self.numKernelsInit*4,self.projectingFactor,self.numKernelsInit*8)
        self.bottleNeck2_1 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck2_2 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,2)
        self.bottleNeck2_3 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck2_4 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,4)
        self.bottleNeck2_5 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck2_6 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,8)
        self.bottleNeck2_7 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck2_8 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,16)
        

        # Third group
        self.bottleNeck3_1 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck3_2 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,2)
        self.bottleNeck3_3 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck3_4 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,4)
        self.bottleNeck3_5 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck3_6 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,8)
        self.bottleNeck3_7 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck3_8 = BottleNeckDownSamplingDilatedConvLast(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*4,16)
        
        #### Decoding path ####
        # Unpooling 1
        self.unpool_0 = nn.MaxUnpool2d(2)
        
        #self.bottleNeck_Up_1_0 = BottleNeckUpSampling(self.numKernelsInit*4, self.projectingFactor,self.numKernelsInit*4 )
        self.bottleNeck_Up_1_0 = BottleNeckUpSampling(self.numKernelsInit*8, self.projectingFactor,self.numKernelsInit*4 ) # If concatenate
        self.PReLU_Up_1 = nn.PReLU()
        
        self.bottleNeck_Up_1_1 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4,self.projectingFactor, 0.1)
        self.bottleNeck_Up_1_2 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit,self.projectingFactor, 0.1)
        
        # Unpooling 2
        self.unpool_1 = nn.MaxUnpool2d(2)
        #self.bottleNeck_Up_2_1 = BottleNeckUpSampling(self.numKernelsInit, self.projectingFactor,self.numKernelsInit )
        self.bottleNeck_Up_2_1 = BottleNeckUpSampling(self.numKernelsInit*2, self.projectingFactor,self.numKernelsInit ) # If concatenate
        self.bottleNeck_Up_2_2 = BottleNeckNormal(self.numKernelsInit,self.numKernelsInit,self.projectingFactor, 0.1)
        self.PReLU_Up_2 = nn.PReLU()
        
        # Unpooling Last
        self.deconv3 = upSampleConv(self.numKernelsInit, self.numKernelsInit)
        #self.conv_out = nn.Sequential(convBatch(nG * 2, nG * 1),
        #                              convBatch(nG * 1, nG * 1))

        self.out_025  = nn.Conv2d(self.numKernelsInit * 8,nout, kernel_size=3, stride=1, padding=1)
        self.out_05   = nn.Conv2d(self.numKernelsInit,nout, kernel_size=3, stride=1, padding=1)
        self.final = nn.Conv2d(self.numKernelsInit, nout, kernel_size=1)
        
    def forward(self,input):
        conv_0 = self.conv0(input)  # This will go as res in deconv path
        maxpool_0,indices_0 = self.maxpool0(input)
        outputInitial = torch.cat((conv_0, maxpool_0), dim=1)

        # First group
        bn1_0, indices_1 = self.bottleNeck1_0(outputInitial)
        bn1_1 = self.bottleNeck1_1(bn1_0)
        bn1_2 = self.bottleNeck1_2(bn1_1)
        bn1_3 = self.bottleNeck1_3(bn1_2)
        bn1_4 = self.bottleNeck1_4(bn1_3)

        # Second group
        bn2_0, indices_2 = self.bottleNeck2_0(bn1_4)
        bn2_1 = self.bottleNeck2_1(bn2_0)
        bn2_2 = self.bottleNeck2_2(bn2_1)
        bn2_3 = self.bottleNeck2_3(bn2_2)
        bn2_4 = self.bottleNeck2_4(bn2_3)
        bn2_5 = self.bottleNeck2_5(bn2_4)
        bn2_6 = self.bottleNeck2_6(bn2_5)
        bn2_7 = self.bottleNeck2_7(bn2_6)
        bn2_8 = self.bottleNeck2_8(bn2_7)
        
         # Third group
        bn3_1 = self.bottleNeck3_1(bn2_8)
        bn3_2 = self.bottleNeck3_2(bn3_1)
        bn3_3 = self.bottleNeck3_3(bn3_2)
        bn3_4 = self.bottleNeck3_4(bn3_3)
        bn3_5 = self.bottleNeck3_5(bn3_4)
        bn3_6 = self.bottleNeck3_6(bn3_5)
        bn3_7 = self.bottleNeck3_7(bn3_6)
        bn3_8 = self.bottleNeck3_8(bn3_7)
        
        
        ##### Deconvolution Path ####
        #  First block #
        unpool_0 = self.unpool_0(bn3_8,indices_2)
        
        #bn_up_1_0 = self.bottleNeck_Up_1_0(unpool_0) # Not concatenate
        bn_up_1_0 = self.bottleNeck_Up_1_0(torch.cat((unpool_0,bn1_4), dim=1)) # concatenate
        
        up_block_1 = self.PReLU_Up_1(unpool_0+bn_up_1_0)
        
        bn_up_1_1 = self.bottleNeck_Up_1_1(up_block_1)
        bn_up_1_2 = self.bottleNeck_Up_1_2(bn_up_1_1)
        
        #  Second block #
        
        unpool_1 = self.unpool_1(bn_up_1_2, indices_1)

        #bn_up_2_1 = self.bottleNeck_Up_2_1(unpool_1) # Not concatenate
        bn_up_2_1 = self.bottleNeck_Up_2_1(torch.cat((unpool_1,outputInitial), dim=1)) # concatenate
        
        bn_up_2_2 = self.bottleNeck_Up_2_2(bn_up_2_1)

        up_block_1 = self.PReLU_Up_2(unpool_1+bn_up_2_2)
 
        unpool_12 = self.deconv3(up_block_1)

        #return F.softmax(self.final(unpool_12), dim=1)
        return self.final(unpool_12)
        

class Reduced_ENet(nn.Module):
    def __init__(self, nin, nout):
        super(Reduced_ENet, self).__init__()
        self.projectingFactor = 4
        self.numKernelsInit = 16
        # Initial
        self.conv0 = nn.Conv2d(nin,15, kernel_size=3, stride=2, padding=1)
        self.maxpool0 = nn.MaxPool2d(2, return_indices = True)

        # First group
        self.bottleNeck1_0 = BottleNeckDownSampling(self.numKernelsInit,self.projectingFactor, self.numKernelsInit*4)
        self.bottleNeck1_1 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)
        self.bottleNeck1_2 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)
        self.bottleNeck1_3 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)
        self.bottleNeck1_4 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)

        # Second group
        self.bottleNeck2_0 = BottleNeckDownSampling(self.numKernelsInit*4,self.projectingFactor,self.numKernelsInit*8)
        self.bottleNeck2_1 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck2_2 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,2)
        self.bottleNeck2_3 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck2_4 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,4)
        self.bottleNeck2_5 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck2_6 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,8)
        self.bottleNeck2_7 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        #self.bottleNeck2_8 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,16)
        

        # Third group
        '''self.bottleNeck3_1 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck3_2 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,2)
        self.bottleNeck3_3 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck3_4 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,4)
        self.bottleNeck3_5 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck3_6 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,8)
        self.bottleNeck3_7 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)'''
        self.bottleNeck3_8 = BottleNeckDownSamplingDilatedConvLast(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*4,16)
        
        #### Decoding path ####
        # Unpooling 1
        self.unpool_0 = nn.MaxUnpool2d(2)
        
        #self.bottleNeck_Up_1_0 = BottleNeckUpSampling(self.numKernelsInit*4, self.projectingFactor,self.numKernelsInit*4 )
        self.bottleNeck_Up_1_0 = BottleNeckUpSampling(self.numKernelsInit*8, self.projectingFactor,self.numKernelsInit*4 ) # If concatenate
        self.PReLU_Up_1 = nn.PReLU()
        
        self.bottleNeck_Up_1_1 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4,self.projectingFactor, 0.1)
        self.bottleNeck_Up_1_2 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit,self.projectingFactor, 0.1)
        
        # Unpooling 2
        self.unpool_1 = nn.MaxUnpool2d(2)
        #self.bottleNeck_Up_2_1 = BottleNeckUpSampling(self.numKernelsInit, self.projectingFactor,self.numKernelsInit )
        self.bottleNeck_Up_2_1 = BottleNeckUpSampling(self.numKernelsInit*2, self.projectingFactor,self.numKernelsInit ) # If concatenate
        self.bottleNeck_Up_2_2 = BottleNeckNormal(self.numKernelsInit,self.numKernelsInit,self.projectingFactor, 0.1)
        self.PReLU_Up_2 = nn.PReLU()
        
        # Unpooling Last
        self.deconv3 = upSampleConv(self.numKernelsInit, self.numKernelsInit)
        #self.conv_out = nn.Sequential(convBatch(nG * 2, nG * 1),
        #                              convBatch(nG * 1, nG * 1))

        self.out_025  = nn.Conv2d(self.numKernelsInit * 8,nout, kernel_size=3, stride=1, padding=1)
        self.out_05   = nn.Conv2d(self.numKernelsInit,nout, kernel_size=3, stride=1, padding=1)
        self.final = nn.Conv2d(self.numKernelsInit, nout, kernel_size=1)
        
    def forward(self,input):
        conv_0 = self.conv0(input)  # This will go as res in deconv path
        maxpool_0,indices_0 = self.maxpool0(input)
        outputInitial = torch.cat((conv_0, maxpool_0), dim=1)

        # First group
        bn1_0, indices_1 = self.bottleNeck1_0(outputInitial)
        bn1_1 = self.bottleNeck1_1(bn1_0)
        bn1_2 = self.bottleNeck1_2(bn1_1)
        bn1_3 = self.bottleNeck1_3(bn1_2)
        bn1_4 = self.bottleNeck1_4(bn1_3)

        # Second group
        bn2_0, indices_2 = self.bottleNeck2_0(bn1_4)
        bn2_1 = self.bottleNeck2_1(bn2_0)
        bn2_2 = self.bottleNeck2_2(bn2_1)
        bn2_3 = self.bottleNeck2_3(bn2_2)
        bn2_4 = self.bottleNeck2_4(bn2_3)
        bn2_5 = self.bottleNeck2_5(bn2_4)
        bn2_6 = self.bottleNeck2_6(bn2_5)
        bn2_7 = self.bottleNeck2_7(bn2_6)
        '''bn2_8 = self.bottleNeck2_8(bn2_7)
        
         # Third group
        bn3_1 = self.bottleNeck3_1(bn2_8)
        bn3_2 = self.bottleNeck3_2(bn3_1)
        bn3_3 = self.bottleNeck3_3(bn3_2)
        bn3_4 = self.bottleNeck3_4(bn3_3)
        bn3_5 = self.bottleNeck3_5(bn3_4)
        bn3_6 = self.bottleNeck3_6(bn3_5)
        bn3_7 = self.bottleNeck3_7(bn3_6)'''
        bn3_8 = self.bottleNeck3_8(bn2_7)
        
        
        ##### Deconvolution Path ####
        #  First block #
        unpool_0 = self.unpool_0(bn3_8,indices_2)
        
        #bn_up_1_0 = self.bottleNeck_Up_1_0(unpool_0) # Not concatenate
        bn_up_1_0 = self.bottleNeck_Up_1_0(torch.cat((unpool_0,bn1_4), dim=1)) # concatenate
        
        up_block_1 = self.PReLU_Up_1(unpool_0+bn_up_1_0)
        
        bn_up_1_1 = self.bottleNeck_Up_1_1(up_block_1)
        bn_up_1_2 = self.bottleNeck_Up_1_2(bn_up_1_1)
        
        #  Second block #
        
        unpool_1 = self.unpool_1(bn_up_1_2, indices_1)

        #bn_up_2_1 = self.bottleNeck_Up_2_1(unpool_1) # Not concatenate
        bn_up_2_1 = self.bottleNeck_Up_2_1(torch.cat((unpool_1,outputInitial), dim=1)) # concatenate
        
        bn_up_2_2 = self.bottleNeck_Up_2_2(bn_up_2_1)

        up_block_1 = self.PReLU_Up_2(unpool_1+bn_up_2_2)
 
        unpool_12 = self.deconv3(up_block_1)

        return F.softmax(self.final(unpool_12), dim=1)


        
class ENet_DeepSupervision(nn.Module):
    def __init__(self, nin, nout):
        super(ENet_DeepSupervision, self).__init__()
        self.projectingFactor = 4
        self.numKernelsInit = 16
        # Initial
        self.conv0 = nn.Conv2d(nin,15, kernel_size=3, stride=2, padding=1)
        self.maxpool0 = nn.MaxPool2d(2, return_indices = True)

        # First group
        self.bottleNeck1_0 = BottleNeckDownSampling(self.numKernelsInit,self.projectingFactor, self.numKernelsInit*4)
        self.bottleNeck1_1 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)
        self.bottleNeck1_2 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)
        self.bottleNeck1_3 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)
        self.bottleNeck1_4 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)

        # Second group
        self.bottleNeck2_0 = BottleNeckDownSampling(self.numKernelsInit*4,self.projectingFactor,self.numKernelsInit*8)
        self.bottleNeck2_1 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck2_2 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,2)
        self.bottleNeck2_3 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck2_4 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,4)
        self.bottleNeck2_5 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck2_6 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,8)
        self.bottleNeck2_7 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck2_8 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,16)
        

        # Third group
        self.bottleNeck3_1 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck3_2 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,2)
        self.bottleNeck3_3 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck3_4 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,4)
        self.bottleNeck3_5 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck3_6 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,8)
        self.bottleNeck3_7 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck3_8 = BottleNeckDownSamplingDilatedConvLast(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*4,16)
        
        #### Decoding path ####
        # Unpooling 1
        self.unpool_0 = nn.MaxUnpool2d(2)
        
        #self.bottleNeck_Up_1_0 = BottleNeckUpSampling(self.numKernelsInit*4, self.projectingFactor,self.numKernelsInit*4 )
        self.bottleNeck_Up_1_0 = BottleNeckUpSampling(self.numKernelsInit*8, self.projectingFactor,self.numKernelsInit*4 ) # If concatenate
        self.PReLU_Up_1 = nn.PReLU()
        
        self.bottleNeck_Up_1_1 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4,self.projectingFactor, 0.1)
        self.bottleNeck_Up_1_2 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit,self.projectingFactor, 0.1)
        
        # Unpooling 2
        self.unpool_1 = nn.MaxUnpool2d(2)
        #self.bottleNeck_Up_2_1 = BottleNeckUpSampling(self.numKernelsInit, self.projectingFactor,self.numKernelsInit )
        self.bottleNeck_Up_2_1 = BottleNeckUpSampling(self.numKernelsInit*2, self.projectingFactor,self.numKernelsInit ) # If concatenate
        self.bottleNeck_Up_2_2 = BottleNeckNormal(self.numKernelsInit,self.numKernelsInit,self.projectingFactor, 0.1)
        self.PReLU_Up_2 = nn.PReLU()
        
        # Unpooling Last
        self.deconv3 = upSampleConv(self.numKernelsInit, self.numKernelsInit)
        #self.conv_out = nn.Sequential(convBatch(nG * 2, nG * 1),
        #                              convBatch(nG * 1, nG * 1))

        self.out_0125  = nn.Conv2d(self.numKernelsInit * 4,nout, kernel_size=3, stride=1, padding=1)
        self.out_025   = nn.Conv2d(self.numKernelsInit    ,nout, kernel_size=3, stride=1, padding=1)
        self.out_05    = nn.Conv2d(self.numKernelsInit    ,nout, kernel_size=3, stride=1, padding=1)
        self.final     = nn.Conv2d(self.numKernelsInit, nout, kernel_size=1)
        
    def forward(self,input):
        conv_0 = self.conv0(input)  # This will go as res in deconv path
        maxpool_0,indices_0 = self.maxpool0(input)
        outputInitial = torch.cat((conv_0, maxpool_0), dim=1)

        # First group
        bn1_0, indices_1 = self.bottleNeck1_0(outputInitial)
        bn1_1 = self.bottleNeck1_1(bn1_0)
        bn1_2 = self.bottleNeck1_2(bn1_1)
        bn1_3 = self.bottleNeck1_3(bn1_2)
        bn1_4 = self.bottleNeck1_4(bn1_3)

        # Second group
        bn2_0, indices_2 = self.bottleNeck2_0(bn1_4)
        bn2_1 = self.bottleNeck2_1(bn2_0)
        bn2_2 = self.bottleNeck2_2(bn2_1)
        bn2_3 = self.bottleNeck2_3(bn2_2)
        bn2_4 = self.bottleNeck2_4(bn2_3)
        bn2_5 = self.bottleNeck2_5(bn2_4)
        bn2_6 = self.bottleNeck2_6(bn2_5)
        bn2_7 = self.bottleNeck2_7(bn2_6)
        bn2_8 = self.bottleNeck2_8(bn2_7)
        
         # Third group
        bn3_1 = self.bottleNeck3_1(bn2_8)
        bn3_2 = self.bottleNeck3_2(bn3_1)
        bn3_3 = self.bottleNeck3_3(bn3_2)
        bn3_4 = self.bottleNeck3_4(bn3_3)
        bn3_5 = self.bottleNeck3_5(bn3_4)
        bn3_6 = self.bottleNeck3_6(bn3_5)
        bn3_7 = self.bottleNeck3_7(bn3_6)
        bn3_8 = self.bottleNeck3_8(bn3_7)
        
        ##### Deconvolution Path ####
        #  First block #
        unpool_0 = self.unpool_0(bn3_8,indices_2)
        # concatenate
        
        bn_up_1_0 = self.bottleNeck_Up_1_0(torch.cat((unpool_0,bn1_4), dim=1))
        
        up_block_1 = self.PReLU_Up_1(unpool_0+bn_up_1_0)
        
        bn_up_1_1 = self.bottleNeck_Up_1_1(up_block_1)
        bn_up_1_2 = self.bottleNeck_Up_1_2(bn_up_1_1)
        
        #  Second block #
        
        unpool_1 = self.unpool_1(bn_up_1_2, indices_1)

        #bn_up_2_1 = self.bottleNeck_Up_2_1(unpool_1)
        bn_up_2_1 = self.bottleNeck_Up_2_1(torch.cat((unpool_1,outputInitial), dim=1))
        bn_up_2_2 = self.bottleNeck_Up_2_2(bn_up_2_1)

        up_block_1 = self.PReLU_Up_2(unpool_1+bn_up_2_2)
 
        unpool_12 = self.deconv3(up_block_1)


        return [F.softmax(self.final(unpool_12), dim=1),
                F.softmax(self.out_05(up_block_1), dim=1),
                F.softmax(self.out_025(bn_up_1_2), dim=1),
                F.softmax(self.out_0125(bn3_8), dim=1)]
                
                
class ENet_MultiTask(nn.Module):
    def __init__(self, nin, nout):
        super(ENet_MultiTask, self).__init__()
        self.projectingFactor = 4
        self.numKernelsInit = 16
        self.imageSize = 256
        # Initial
        self.conv0 = nn.Conv2d(nin,15, kernel_size=3, stride=2, padding=1)
        self.maxpool0 = nn.MaxPool2d(2, return_indices = True)

        # First group
        self.bottleNeck1_0 = BottleNeckDownSampling(self.numKernelsInit,self.projectingFactor, self.numKernelsInit*4)
        self.bottleNeck1_1 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)
        self.bottleNeck1_2 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)
        self.bottleNeck1_3 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)
        self.bottleNeck1_4 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4, self.projectingFactor, 0.01)

        # Second group
        self.bottleNeck2_0 = BottleNeckDownSampling(self.numKernelsInit*4,self.projectingFactor,self.numKernelsInit*8)
        self.bottleNeck2_1 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck2_2 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,2)
        self.bottleNeck2_3 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck2_4 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,4)
        self.bottleNeck2_5 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck2_6 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,8)
        self.bottleNeck2_7 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck2_8 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,16)
        

        # Third group
        self.bottleNeck3_1 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck3_2 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,2)
        self.bottleNeck3_3 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck3_4 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,4)
        self.bottleNeck3_5 = BottleNeckNormal(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck3_6 = BottleNeckDownSamplingDilatedConv(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*8,8)
        self.bottleNeck3_7 = BottleNeckNormal_Asym(self.numKernelsInit*8,self.numKernelsInit*8, self.projectingFactor, 0.1)
        self.bottleNeck3_8 = BottleNeckDownSamplingDilatedConvLast(self.numKernelsInit*8,self.projectingFactor, self.numKernelsInit*4,16)
        
        #### Decoding path ####
        # Unpooling 1
        self.unpool_0 = nn.MaxUnpool2d(2)
        
        #self.bottleNeck_Up_1_0 = BottleNeckUpSampling(self.numKernelsInit*4, self.projectingFactor,self.numKernelsInit*4 )
        self.bottleNeck_Up_1_0 = BottleNeckUpSampling(self.numKernelsInit*8, self.projectingFactor,self.numKernelsInit*4 ) # If concatenate
        self.PReLU_Up_1 = nn.PReLU()
        
        self.bottleNeck_Up_1_1 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit*4,self.projectingFactor, 0.1)
        self.bottleNeck_Up_1_2 = BottleNeckNormal(self.numKernelsInit*4,self.numKernelsInit,self.projectingFactor, 0.1)
        
        # Unpooling 2
        self.unpool_1 = nn.MaxUnpool2d(2)
        #self.bottleNeck_Up_2_1 = BottleNeckUpSampling(self.numKernelsInit, self.projectingFactor,self.numKernelsInit )
        self.bottleNeck_Up_2_1 = BottleNeckUpSampling(self.numKernelsInit*2, self.projectingFactor,self.numKernelsInit ) # If concatenate
        self.bottleNeck_Up_2_2 = BottleNeckNormal(self.numKernelsInit,self.numKernelsInit,self.projectingFactor, 0.1)
        self.PReLU_Up_2 = nn.PReLU()
        
        # Unpooling Last
        self.deconv3 = upSampleConv(self.numKernelsInit, self.numKernelsInit)
        #self.conv_out = nn.Sequential(convBatch(nG * 2, nG * 1),
        #                              convBatch(nG * 1, nG * 1))
        
        self.final     = nn.Conv2d(self.numKernelsInit, nout, kernel_size=1)

        #####  For the regression path #####
        numFeatures = int(self.imageSize*self.imageSize/64*self.numKernelsInit*4)
        hiddenUnits = 2048
        self.classBlock = classificationBlock(numFeatures,hiddenUnits,3)
        
    def forward(self,input):
        conv_0 = self.conv0(input)  # This will go as res in deconv path
        maxpool_0,indices_0 = self.maxpool0(input)
        outputInitial = torch.cat((conv_0, maxpool_0), dim=1)

        # First group
        bn1_0, indices_1 = self.bottleNeck1_0(outputInitial)
        bn1_1 = self.bottleNeck1_1(bn1_0)
        bn1_2 = self.bottleNeck1_2(bn1_1)
        bn1_3 = self.bottleNeck1_3(bn1_2)
        bn1_4 = self.bottleNeck1_4(bn1_3)

        # Second group
        bn2_0, indices_2 = self.bottleNeck2_0(bn1_4)
        bn2_1 = self.bottleNeck2_1(bn2_0)
        bn2_2 = self.bottleNeck2_2(bn2_1)
        bn2_3 = self.bottleNeck2_3(bn2_2)
        bn2_4 = self.bottleNeck2_4(bn2_3)
        bn2_5 = self.bottleNeck2_5(bn2_4)
        bn2_6 = self.bottleNeck2_6(bn2_5)
        bn2_7 = self.bottleNeck2_7(bn2_6)
        bn2_8 = self.bottleNeck2_8(bn2_7)
        
         # Third group
        bn3_1 = self.bottleNeck3_1(bn2_8)
        bn3_2 = self.bottleNeck3_2(bn3_1)
        bn3_3 = self.bottleNeck3_3(bn3_2)
        bn3_4 = self.bottleNeck3_4(bn3_3)
        bn3_5 = self.bottleNeck3_5(bn3_4)
        bn3_6 = self.bottleNeck3_6(bn3_5)
        bn3_7 = self.bottleNeck3_7(bn3_6)
        bn3_8 = self.bottleNeck3_8(bn3_7)
        
        
        featuresRegressionPath = bn3_8.view(bn3_8.shape[0],int(bn3_8.numel()/bn3_8.shape[0])) # Flatten to (batchSize, numElemsPerSample)
        regressionOut = self.classBlock(featuresRegressionPath)
        
        ##### Deconvolution Path ####
        #  First block #
        unpool_0 = self.unpool_0(bn3_8,indices_2)
        # concatenate
        
        bn_up_1_0 = self.bottleNeck_Up_1_0(torch.cat((unpool_0,bn1_4), dim=1))
        
        up_block_1 = self.PReLU_Up_1(unpool_0+bn_up_1_0)
        
        bn_up_1_1 = self.bottleNeck_Up_1_1(up_block_1)
        bn_up_1_2 = self.bottleNeck_Up_1_2(bn_up_1_1)
        
        #  Second block #
        
        unpool_1 = self.unpool_1(bn_up_1_2, indices_1)

        #bn_up_2_1 = self.bottleNeck_Up_2_1(unpool_1)
        bn_up_2_1 = self.bottleNeck_Up_2_1(torch.cat((unpool_1,outputInitial), dim=1))
        bn_up_2_2 = self.bottleNeck_Up_2_2(bn_up_2_1)

        up_block_1 = self.PReLU_Up_2(unpool_1+bn_up_2_2)
 
        unpool_12 = self.deconv3(up_block_1)

        return [F.softmax(self.final(unpool_12), dim=1),
                regressionOut]
        
                
       
