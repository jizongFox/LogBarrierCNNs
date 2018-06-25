import numpy as np
import torch, torch.nn as nn
from criterion import CrossEntropyLoss2d
from medpy.io import load, header
from medpy.graphcut.generate import graph_from_labels, graph_from_voxels
from medpy.graphcut.energy_voxel import boundary_difference_exponential, boundary_maximum_exponential
from medpy.graphcut.energy_voxel import regional_probability_map
import torch.nn.functional as F
import matplotlib.pyplot as plt

def graphcut3D(im, hm, lamda, sigma, eps=1e-10, pWeightMax=None):
    '''
    :param im: image
    :param hm: heatmap
    :param lamda:
    :param sigma:
    :param eps:
    :param pWeightMax:
    :return:
    '''
    hm=F.softmax(hm,dim=1).squeeze(0).cpu().data.numpy()
    pf = np.log(hm[1,:,:] + eps) - np.log(hm[0,:,:]+eps)
    pb = np.log(hm[0, :,:] + eps) - np.log(hm[1,:,:]+eps)
    # pb = np.zeros(im.shape)

    # Maximum weight for the pixels predicted foreground by the CNN
    pixelsFG_weight_max = hm.argmax(axis=0)
    pixelsBG_weight_max = np.zeros(im.shape)
    pixelsBG_weight_max[20:30][20:30]=1
    # pixelsBG_weight_max = 1- pixelsFG_weight_max
    # background

    if pWeightMax:
        pixelsFG_weight_max[hm[:,:,1,:] < pWeightMax] = 0

    # for i in range(pixelsFG_weight_max.shape[2]):
    #     label_bin = pixelsFG_weight_max[:,:,i].copy()
    #     label_rp = skimage.measure.label(label_bin)
    #     rp = skimage.measure.regionprops(label_rp)

    #     rp = max(rp, key=lambda r: r.area)

    #     pixelsFG_weight_max[:,:,i] = np.zeros((im.shape[0], im.shape[1]))
    #     for coo in rp.coords:
    #         pixelsFG_weight_max[coo[0], coo[1],i] = 1



    # pixelsFG_weight_max[pixelsBG_weight_max == 1] = 0
    # foreground

    im_boundary_term = im.squeeze().data.cpu().numpy()

    gcgraph = graph_from_voxels(
        pixelsFG_weight_max,
        # pf,
        # pb,
                                    pixelsBG_weight_max,
                                    #regional_term = regional_weight_map,
                                    # regional_term = regional_probability_map,
                                    # regional_term_args = (hm[:,:,:], lamda),
                                    #regional_term_args = (pb,pf,alpha1),
                                    boundary_term = boundary_difference_exponential,
                                    boundary_term_args = (im_boundary_term, sigma, False))

    maxflow = gcgraph.maxflow()

    result_image_data = np.zeros(im_boundary_term.size, dtype=np.bool)
    for idx in range(len(result_image_data)):
        result_image_data[idx] = 0 if gcgraph.termtype.SINK == gcgraph.what_segment(idx) else 1

    result_image_data = result_image_data.reshape(im_boundary_term.shape)

    # plt.imshow(result_image_data);plt.show()

    # We only keep the biggest connexe composante
    # for i in range(result_image_data.shape[2]):
    #     label_bin = result_image_data[:,:,i].copy()
    #     label_rp = skimage.measure.label(label_bin)
    #     rp = skimage.measure.regionprops(label_rp)
    #
    #     if rp == []:
    #         result_image_data[:,:,i] = np.zeros((im.shape[:2]))
    #         continue
    #
    #     rp = max(rp, key=lambda r: r.area)
    #
    #     result_image_data[:,:,i] = np.zeros((im.shape[0], im.shape[1]))
    #     for coo in rp.coords:
    #         result_image_data[coo[0], coo[1],i] = 1
    return result_image_data


def update_theta(f_theta_labeled,f_theta_unlabeled,gamma,s, u, v):
    global u_r,u_s,net
    global labeled_img, labeled_mask, labeled_weak_mask, unlabeled_img, unlabeled_mask
    optimiser = torch.optim.Adam(net.parameters(),lr=1e-3)
    criterion = CrossEntropyLoss2d()
    for i in xrange(10):

        loss_labeled = criterion(f_theta_labeled,labeled_mask.squeeze(1))
        loss_unlabeled = u_r * (f_theta_unlabeled - gamma.float() + torch.Tensor(u)).norm(p=2)+\
                         u_s * (f_theta_unlabeled-s+v).norm(p=2)
        loss = loss_labeled + loss_unlabeled
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        f_theta_labeled = net(labeled_img)  # shape b,c,w,h
        f_theta_unlabeled = net(unlabeled_img)

    return f_theta_labeled,f_theta_unlabeled

