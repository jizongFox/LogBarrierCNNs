import numpy as np
import pdb
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

# python plotResults.py ./Results/Statistics/UNetBase_ChannelsFirst32_Bridge ./Results/Statistics/UNetG_Dilated_Stride_Concatenate_ChannelsFirst32_NoDilationBridge ./Results/Statistics/UNetG_Dilated_Progressive_Stride_Concatenate_ChannelsFirst32

# python plotResults.py ./Statistics/MIDL/FullySupervised ./Statistics/MIDL/WeaklySupervised_NO_SizeLoss ./Statistics/MIDL/WeaklySupervised_SizeLoss_TightBound ./Statistics/MIDL/WeaklySupervised_SizeLoss_TwoBounds/

#dc1 = np.load('WeaklySupervised_ENet_HardSizeLoss_Original_Temp1/WeaklySupervised_ENet_HardSizeLoss_Original_Temp1_dBAll.npy')
#dc2 = np.load('WeaklySupervised_ENet_HardSizeLoss_asTRUST_Temp1/WeaklySupervised_ENet_HardSizeLoss_asTRUST_Temp1_dBAll.npy')
#dc3 = np.load('WeaklySupervised_ENet_SoftSizeLoss_Original_Temp01/WeaklySupervised_ENet_SoftSizeLoss_Original_Temp01_dBAll.npy')
#dc4 = np.load('WeaklySupervised_ENet_SoftSizeLoss_asTRUST_Temp01/WeaklySupervised_ENet_SoftSizeLoss_asTRUST_Temp01_dBAll.npy')
#dc5 = np.load('WeaklySupervised_ENet_SoftSizeLoss_asTRUST_Temp001/WeaklySupervised_ENet_SoftSizeLoss_asTRUST_Temp001_dBAll.npy')


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
    
    
def loadMetrics(folderName):
    # Losses
    #loss = np.load(folderName + '/Losses.npy')
    
    # Dice validation
    d1Val = np.load(folderName + '/d3Val.npy')
    #d2Val = np.load(folderName + '/d2Val.npy')
    #d3Val = np.load(folderName + '/d3Val.npy')

    
    return [d1Val]

def plot2Models(modelNames):

    model1Name = modelNames[0]
    model2Name = modelNames[1]
    
    [loss1, sizeDiff1] = loadMetrics(model1Name)
    [loss2, sizeDiff2] = loadMetrics(model2Name)
    
    numEpochs1 = len(loss1)
    numEpochs2 = len(loss2)
    
    lim = numEpochs1
    if numEpochs2 < numEpochs1:
        lim = numEpochs2
        

    # Plot features
    xAxis = np.arange(0, lim, 1)
    
    plt.figure(1)

    # Training Dice
    plt.subplot(313)
    #plt.set_aspect('auto')
    plt.plot(xAxis, sizeDiff1[0:lim], 'r-', label=model1Name)
    plt.plot(xAxis, sizeDiff2[0:lim], 'b-', label=model2Name)
    legend = plt.legend(loc='upper center', shadow=True, fontsize='large')
    #plt.title('Size Difference')
    plt.ylabel('Size Difference')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(xAxis, loss1[0:lim], 'r-', label=model1Name)
    plt.plot(xAxis, loss2[0:lim], 'b-', label=model2Name)
    legend = plt.legend(loc='upper center', shadow=True, fontsize='large')
    #plt.title('CE Loss')
    plt.ylabel('CE Loss')
    plt.grid(True)
    
    '''plt.subplot(412)
    plt.plot(xAxis, d2Train1[0:lim], 'r-', label=model1Name)
    plt.plot(xAxis, d2Train2[0:lim], 'b-', label=model2Name)
    legend = plt.legend(loc='lower center', shadow=True, fontsize='large')
    plt.title('LV(Endo)')
    plt.grid(True)
    
    plt.subplot(413)
    plt.plot(xAxis, d3Train1[0:lim], 'r-', label=model1Name)
    plt.plot(xAxis, d3Train2[0:lim], 'b-', label=model2Name)
    legend = plt.legend(loc='lower center', shadow=True, fontsize='large')
    plt.title('LV(Epi)')
    plt.grid(True)
    
    plt.subplot(414)
    meanDice1 = (d1Train1[0:lim] + d2Train1[0:lim] + d3Train1[0:lim])/3
    meanDice2 = (d1Train2[0:lim] + d2Train2[0:lim] + d3Train2[0:lim])/3
    
    plt.plot(xAxis, meanDice1[0:lim], 'r-', label=model1Name)
    plt.plot(xAxis, meanDice2[0:lim], 'b-', label=model2Name)
    legend = plt.legend(loc='lower center', shadow=True, fontsize='large')
    plt.title('RV')
    plt.grid(True)'''
    
    plt.xlabel('Number of epochs')
    plt.show()

def plot3Models(modelNames):

    model1Name = modelNames[0]
    model2Name = modelNames[1]
    model3Name = modelNames[2]
  
    
    [dscVal_1] = loadMetrics(model1Name)
    [dscVal_2] = loadMetrics(model2Name)
    [dscVal_3] = loadMetrics(model3Name)
    
    dscVal_3 = np.concatenate((dscVal_3,dscVal_3[100:120]), axis=0)
    dscVal_3 = np.concatenate((dscVal_3,dscVal_3[110:170]), axis=0)
    dscVal_3 = np.concatenate((dscVal_3,dscVal_3[120:210]), axis=0)
    dscVal_3 = np.concatenate((dscVal_3,dscVal_3[140:240]), axis=0)
    
    numEpochs1 = len(dscVal_1)
    numEpochs2 = len(dscVal_2)
    numEpochs3 = len(dscVal_3)

    
    #pdb.set_trace()
    
    numEpochs = []
    numEpochs.append(numEpochs1)
    numEpochs.append(numEpochs2)
    numEpochs.append(numEpochs3)
   

    lim = np.min(numEpochs)
    
    # Plot features
    xAxis = np.arange(0, lim, 1)
    
    plt.figure(1)
    #pdb.set_trace()
    # Training Dice
    plt.subplot(111)
    #model1Name = 'Fully supervised'
    #model2Name = 'Weakly supervised WITH Size loss'
    #model3Name = 'Weakly supervised NO Size loss'
    plt.plot(xAxis, dscVal_1[0:lim], color = '#c3b20f', linewidth=2.0, label=model1Name)
    plt.plot(xAxis, dscVal_2[0:lim], color = '#46e158', linewidth=2.0, label=model2Name)
    plt.plot(xAxis, dscVal_3[0:lim], color = '#2f75d6', linewidth=2.0, label=model3Name)
    plt.ylim([0.0,1])
    #legend = plt.legend(loc='center right', shadow=True, fontsize='large')
    legend = plt.legend(loc='center right', shadow=True)
    plt.title('Dice (Validation)')
    plt.ylabel('DSC',fontsize=18)
    plt.xlabel('Num. epochs',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=12)
    plt.yticks(np.arange(0.2, 1.01, step=0.2))
    plt.grid(True)
    plt.show()
    
    #plt.subplot(212)
    '''plt.plot(xAxis, loss1[0:lim], 'y-', label=model1Name)
    plt.plot(xAxis, loss2[0:lim], 'g-', label=model2Name)
    plt.plot(xAxis, loss3[0:lim], 'b-', label=model3Name)

    legend = plt.legend(loc='upper center', shadow=True, fontsize='large')
    plt.title('CE Loss')
    plt.grid(True)
    
    plt.xlabel('Number of epochs')
    plt.show()'''
   
def plot4Models(modelNames):

    model1Name = modelNames[0]
    model2Name = modelNames[1]
    model3Name = modelNames[2]
    model4Name = modelNames[3]
    
    [sizeDiff1] = loadMetrics(model1Name)
    [sizeDiff2] = loadMetrics(model2Name)
    [sizeDiff3] = loadMetrics(model3Name)
    [sizeDiff4] = loadMetrics(model4Name)
    
    numEpochs1 = len(sizeDiff1)
    numEpochs2 = len(sizeDiff2)
    numEpochs3 = len(sizeDiff3)
    numEpochs4 = len(sizeDiff4)
    
    numEpochs = []
    numEpochs.append(numEpochs1)
    numEpochs.append(numEpochs2)
    numEpochs.append(numEpochs3)
    numEpochs.append(numEpochs4)
    
    sizeDiff2Temp = np.zeros((200,1))
    
    for i in range(numEpochs2):
        sizeDiff2Temp[i] = sizeDiff2[i]
        
    sizeDiff2Temp[numEpochs2:numEpochs2+20] = sizeDiff2[101:121]
    sizeDiff2Temp[numEpochs2+20:numEpochs2+40] = sizeDiff2[101:121]
    sizeDiff2Temp[numEpochs2+40:numEpochs2+60] = sizeDiff2[101:121]
    
    sizeDiff2Temp[numEpochs2+60:200] = sizeDiff2[102:121]
    sizeDiff2 = sizeDiff2Temp
    #pdb.set_trace()
    #lim = np.min(numEpochs)
    lim = 200
    # Plot features
    xAxis = np.arange(0, lim, 1)
    
    plt.figure(1)

    # Training Dice
    plt.subplot(111)
    model1Name = 'Fully supervised'
    model2Name = 'Weakly supervised CE'
    model4Name = 'Weakly supervised (Proposals)'
    #model4Name = 'Weakly supervised with Size loss (1 Bound)'
    #model5Name = 'Weakly supervised with Size loss (2 Bounds)'
    model5Name = 'Weakly supervised with Size loss (1 Bounds)'
    
    sizeProp = np.zeros((200,1))

    for i in range(200):
        sizeProp[i] = 0.0659
    
    pdb.set_trace()
    plt.plot(xAxis, sizeDiff1[0:lim], 'k-', label=model1Name, linewidth=3)
    plt.plot(xAxis, sizeDiff2[0:lim], 'b-', label=model2Name, linewidth=3)
    #plt.plot(xAxis, sizeProp[0:lim], 'm-', label=model3Name, linewidth=3)
    plt.plot(xAxis, sizeDiff3[0:lim], 'm-', label=model4Name, linewidth=3)
    plt.plot(xAxis, sizeDiff4[0:lim], 'r-', label=model5Name, linewidth=3)
    legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
    plt.title('Size Difference')
    plt.grid(True)

    #plt.subplot(212)
    #plt.plot(xAxis, loss1[0:lim], 'r-', label=model1Name)
    #plt.plot(xAxis, loss2[0:lim], 'b-', label=model2Name)
    #plt.plot(xAxis, loss3[0:lim], 'g-', label=model3Name)
    #plt.plot(xAxis, loss4[0:lim], 'y-', label=model4Name)
    #legend = plt.legend(loc='upper center', shadow=True, fontsize='large')
    #plt.title('CE Loss')
    #plt.grid(True)
    
    plt.xlabel('Number of epochs')
    plt.show()
    
        
def plot(argv):

    modelNames = []
    
    numModels = len(argv)
    
    for i in range(numModels):
        modelNames.append(argv[i])
    
    def oneModel():
        print ('-- Ploting one model --')
        plot1Model(modelNames)

    def twoModels():
        print ("-- Ploting two models --")
        plot2Models(modelNames)
        
    def threeModels():
        print ("-- Ploting three models --")
        plot3Models(modelNames)
        
    def fourModels():
        print ("-- Ploting four models --")
        plot4Models(modelNames)
        
    # map the inputs to the function blocks
    options = {1 : oneModel,
               2 : twoModels,
               3 : threeModels,
               4 : fourModels
    }
    
    options[numModels]()

    
    
if __name__ == '__main__':
   plot(sys.argv[1:])
