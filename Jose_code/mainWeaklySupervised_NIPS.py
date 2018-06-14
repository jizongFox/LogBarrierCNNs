from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar
import torch.nn as nn
import medicalDataLoader
from UNet import *
from utils import *
import pdb
import torch

from ENet import *
from FCN import *

from losses import *
from losses_Log_Barrier import *

def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def getOneHot_Encoded_Segmentation(batch):
    backgroundVal = 0
    foregroundVal = 1.0
    oneHotLabels = torch.cat((batch == backgroundVal, batch == foregroundVal), dim=1)
    return oneHotLabels.float()

def runTraining():
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

    train_set = medicalDataLoader.MedicalImageDataset('train',
                                                      root_dir,
                                                      transform=transform,
                                                      mask_transform=mask_transform,
                                                      augment=False,
                                                      equalize=False)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=5,
                              shuffle=False)

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
    
                                                                            
    # Getting label statistics

    #### To Create weak labels ###
    '''for j, data in enumerate(train_loader):
            image, labels, img_names = data
            backgroundVal = 0
            foregroundVal = 1.0
   
            oneHotLabels = (labels == foregroundVal)

            gt_numpy = (oneHotLabels.numpy()).reshape((256,256))
            gt_eroded = gt_numpy
        
            if (gt_numpy.sum()>0):
                # Erode it
                struct2 = ndimage.generate_binary_structure(2, 3)
                gt_eroded = ndimage.binary_erosion(gt_numpy, structure=struct2,iterations=10).astype(gt_numpy.dtype)
        
                # To be sure that we do not delete the Weakly annoated label
                if (gt_eroded.sum() == 0):
                    gt_eroded = ndimage.binary_erosion(gt_numpy, structure=struct2,iterations=7).astype(gt_numpy.dtype)

                if (gt_eroded.sum() == 0):
                    gt_eroded = ndimage.binary_erosion(gt_numpy, structure=struct2,iterations=3).astype(gt_numpy.dtype)
                
            gt_eroded_Torch = torch.from_numpy(gt_eroded.reshape((1,256,256))).float()

            path = 'WeaklyAnnotations'
            if not os.path.exists(path):
                os.makedirs(path)
        
            name = img_names[0].split('../Corstem/ACDC-2D-All/train/Img/' )
            name = name[1]
            torchvision.utils.save_image(gt_eroded_Torch, os.path.join(path, name), nrow=1,   padding=2,  normalize=False, range=None, scale_each=False,  pad_value=0)
            '''
            
    print("~~~~~~~~~~~ Getting statistics ~~~~~~~~~~")
    LV_Sizes_Sys = []
    LV_Sizes_Dyas = []
    names = []
    '''for j, data in enumerate(train_loader):
            image, labels, weak_labels, img_names = data
            backgroundVal = 0
            foregroundVal = 1.0
            names.append(img_names)
            oneHotLabels = (labels == foregroundVal)

            if (oneHotLabels.sum() > 0):
                str_split = img_names[0].split('_')
                str_split = str_split[1]
                cycle = int(str_split)
                if cycle == 1:
                    LV_Sizes_Sys.append(oneHotLabels.sum())
                else:
                    LV_Sizes_Dyas.append(oneHotLabels.sum())
    
    minVal_Sys = np.min(LV_Sizes_Sys)*0.9
    maxVal_Sys = np.max(LV_Sizes_Sys)*1.1
    
    minVal_Dyas = np.min(LV_Sizes_Dyas)*0.9
    maxVal_Dyas = np.max(LV_Sizes_Dyas)*1.1
    
    minSys = 142 # = 158*0.9
    maxSys = 2339 # = 2127*1.1
    
    minDyas = 80 # 89*0.9
    maxDyas = 1868 # 1698*1.1
    '''
    minVal = 97.9
    #minVal = np.min(LV_Sizes_Sys)
    maxVal = 1722.6
    #maxVal = 10000
    #maxVal = maxVal_Dyas
    #pdb.set_trace()
    # For LogBarrier

    t = 1.0
    mu=1.001

    currentDice = 0.0
#    for i in range(200):
#        t = t*mu
#        print(' t: {}'.format(t))  


    
    
    # Initialize
    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")
    num_classes = 2

    # ENet
    netG = ENet(1,num_classes)
    #netG = FCN8s(num_classes)
    #netG = UNetG_Dilated(1,16,4)

    netG.apply(weights_init)
    softMax = nn.Softmax()
    Dice_loss = computeDiceOneHotBinary()
    
    '''BCE_loss = nn.BCELoss()
    CE_loss = nn.CrossEntropyLoss()
    Dice_loss = computeDiceOneHot()
    MSE_loss = torch.nn.MSELoss()  # this is for regression mean squared loss
    '''

    #modelName = 'WeaklySupervised_LogBarrier_ScheduleT_mu1025_DoubleSoftMax'
    #modelName = 'WeaklySupervised_LogBarrier_ScheduleT_tInit_5_mu101_Weighted_numAnnotatedPixels_DerivateCorrected_WeightLogBar_01'
    #modelName = 'WeaklySupervised_LogBarrier_ScheduleT_tInit_5_mu1005_Weighted_numAnnotatedPixels_DerivateCorrected_WeightLogBar_1'
    #modelName = 'WeaklySupervised_LogBarrier_ScheduleT_tInit_5_mu1005_Weighted_numAnnotatedPixels_DerivateCorrected_WeightLogBar_NoWeighted'
    modelName = 'WeaklySupervised_LogBarrier_NIPS3'
    #modelName = 'WeaklySupervised_LogBarrier_ScheduleT_tInit_5_mu101_Weighted_001_DoubleSoftMax'
    #modelName = 'WeaklySupervised_NaiveLoss'

    print(' ModelName: {}'.format(modelName))
       
    #CE_loss_weakly_numpy_OneHot = myCE_Loss_Weakly_numpy_OneHot()
    CE_loss_weakly_numpy_OneHot = myCE_Loss_Weakly_numpy_OneHot_SoftMaxPyTorch()
    #sizeLoss = mySize_Loss_numpy()
    #sizeLoss = mySize_Loss_numpy_SoftMaxPyTorch()
    #sizeLoss_LOG_BARRIER_OneBound = mySize_Loss_LOG_BARRIER_ONE_BOUND()
    sizeLoss_LOG_BARRIER_Twobounds = mySize_Loss_LOG_BARRIER_TWO_BOUNDS()
    
    if torch.cuda.is_available():
        netG.cuda()
        softMax.cuda()
        Dice_loss.cuda()
    
    '''modelName = 'WeaklySupervised_ENet_HardSizeLoss_NewValues'
    
    try:
        netG = torch.load('./model/Best_' + modelName+'.pkl')
        print("--------model restored--------")
    except:
        print("--------model not restored--------")
        pass'''
        
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
    violConstraintsNeg = []
    violConstraintsPos = []
    violConstraintsTotal = []
    violConstraintsNeg_Distance = []
    violConstraintsPos_Distance = []

    predSizeArr_train = []
    predSizeArr_val = []
    targetSizeArr_train = []
    targetSizeArr_val = []

        
    annotatedPixels = 0
    totalPixels = 0
    
    print(" ~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    print(' --------- Params: ---------')
    print(' - Lower bound: {}'.format(minVal))
    print(' - Upper bound: {}'.format(maxVal))
    print(' - t (logBarrier): {}'.format(t))
    for i in range(epoch):
        netG.train()
        lossVal = []
        lossVal1 = []

        totalImages = len(train_loader)
        #d1, sizeGT, sizePred = inference(netG, 0.1, val_loader, batch_size, i, False, modelName, minVal, maxVal)
        predSizeArrBatches_train = []
        #predSizeArrBatches_val = []
        targetSizeArrBatches_train = []
        #targetSizeArrBatches_val = []
        
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

            # ----- To compute the predicted and target size ------
            predSize = torch.sum((predClass_y[:,1,:,:] > 0.5).type(torch.FloatTensor))
            predSizeNumpy = predSize.cpu().data.numpy()

            LV_target = (labels == 1).type(torch.FloatTensor)
            targetSize = torch.sum(LV_target)
            targetSizeNumpy = targetSize # targetSize.cpu().data.numpy()[0]

            predSizeArrBatches_train.append(predSizeNumpy)
            targetSizeArrBatches_train.append(targetSizeNumpy)
            # ---------------------------------------------- #

            
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

            t_logB = torch.FloatTensor(1)
            t_logB.fill_(np.int64(t).item())
            
            #sizeLoss_val = sizeLoss(segmentation_prediction, Segmentation_planes, Variable(minSize), Variable( maxSize))
            #sizeLoss_val = sizeLoss(predClass_y, Segmentation_planes, Variable(minSize), Variable( maxSize))
            
            #sizeLoss_val = sizeLoss_LOG_BARRIER_OneBound(predClass_y, Segmentation_planes, Variable( maxSize), Variable(t_logB))
            sizeLoss_val = sizeLoss_LOG_BARRIER_Twobounds(predClass_y, Segmentation_planes, Variable( minSize), Variable( maxSize), Variable(t_logB))
            #CE_lossG = CE_loss(segmentation_prediction, Segmentation_class)
        
            # Dice loss (ONLY USED TO COMPUTE THE DICE. This DICE loss version does not work)
            DicesN, DicesB = Dice_loss(segmentation_prediction_ones, Segmentation_planes)
            DiceN = DicesToDice(DicesN)
            DiceB = DicesToDice(DicesB)
            
            Dice_score = (DiceB + DiceN ) / 2
           
           
            lossG = lossCE_numpy + sizeLoss_val 
            #lossG = lossCE_numpy 

   
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


        predSizeArr_train.append(predSizeArrBatches_train)
        targetSizeArr_train.append(targetSizeArrBatches_train)

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
        #modelName = 'WeaklySupervised_LOGBARRIER_1_TwoTightBounds'

        #Losses.append(np.mean(lossVal))
        Losses.append(np.mean(lossVal1))

        #d1, percViol, violCases = inference(netG, temperature, val_loader, batch_size, i, deepSupervision)
        d1, targetSizeArrBatches_val, predSizeArrBatches_val = inference(netG, temperature, val_loader, batch_size, i, deepSupervision, modelName, minVal, maxVal)

        
        predSizeArr_val.append(predSizeArrBatches_val)
        targetSizeArr_val.append(targetSizeArrBatches_val)
                            
        
        dBAll.append(d1)
        #percV.append(percViol)
        
        
        [violPercNeg,
        violPercPos,
        violDistanceNeg,
        violDistanceNeg_min,
        violDistanceNeg_max,
        violDistancePos,
        violDistancePos_min,
        violDistancePos_max]  = analyzeViolationContraints(targetSizeArrBatches_val,predSizeArrBatches_val, minVal, maxVal)
        
        violConstraintsNeg.append(violPercPos)
        violConstraintsPos.append(violPercNeg)
        violConstraintsTotal.append(violPercPos+violPercNeg)
        violConstraintsNeg_Distance.append(violDistanceNeg)
        violConstraintsPos_Distance.append(violDistancePos)

             
        #dWAll.append(d2)
        #dTAll.append(d3)

        
        directory = 'Results/Statistics/MIDL/' + modelName
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(os.path.join(directory, modelName + '_Losses.npy'), Losses)
        
        np.save(os.path.join(directory, modelName + '_dBAll.npy'), dBAll)
        
        np.save(os.path.join(directory, modelName + '_percViolated_Neg.npy'), violConstraintsNeg)
        
        np.save(os.path.join(directory, modelName + '_percViolated_Pos.npy'), violConstraintsPos)
        
        np.save(os.path.join(directory, modelName + '_percViolated_Total.npy'), violConstraintsTotal)
        
        np.save(os.path.join(directory, modelName + '_diffViolated_Neg.npy'), violConstraintsNeg_Distance )
        
        np.save(os.path.join(directory, modelName + '_diffViolated_Pos.npy'), violConstraintsPos_Distance)
        
        np.save(os.path.join(directory, modelName + '_predSizes_train.npy'), predSizeArr_train)
        np.save(os.path.join(directory, modelName + '_predSizes_val.npy'), predSizeArr_val)
        np.save(os.path.join(directory, modelName + '_targetSizes_train.npy'), targetSizeArr_train)
        np.save(os.path.join(directory, modelName + '_targetSizes_val.npy'), targetSizeArr_val)
        
        
        t = t*mu
        print(' t: {}'.format(t))
            
        #print("[val] DSC: (1): {:.4f} ".format(d1[0]))
        print(" [VAL] DSC: (1): {:.4f} ".format(d1))
        print(' [VAL] NEGATIVE: Constrained violated in {:.4f} % of images ( Mean diff = {})'.format(violPercNeg,violDistanceNeg))
        print(' [VAL] POSITIVE: Constrained violated in {:.4f} % of images ( Mean diff = {})'.format(violPercPos,violDistancePos))
        print(' [VAL] TOTAL: Constrained violated in {:.4f} % of images '.format(violPercNeg + violPercPos))
        #saveImagesSegmentation(netG, val_loader_save_imagesPng, batch_size_val_savePng, i, 'test', False)
        
        #if (d1[0]>0.80):
        if (d1>BestDice):
             BestDice = d1
             if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                    
             torch.save(netG, os.path.join(model_dir, "Best_" + modelName + ".pkl"))
             saveImages(netG, val_loader_save_imagesPng, batch_size_val_savePng, i, modelName, deepSupervision) 
             
             
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
    runTraining()
