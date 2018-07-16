import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import medicalDataLoader
from criterion import CrossEntropyLoss2d
from enet import Enet
from utils import pred2segmentation, Colorize
from visualize import Dashboard
from  utils import show_image_mask
from pretrain_network import pretrain

board_image = Dashboard(server='http://turing.livia.etsmtl.ca', env="ADMM_image")
board_loss = Dashboard(server='http://turing.livia.etsmtl.ca', env="ADMM_loss")

use_gpu = True
# device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
device =torch.device('cuda')

batch_size = 1
batch_size_val = 1
num_workers = 0
lr = 0.001
max_epoch = 100
root_dir = '../ACDC-2D-All'
model_dir = 'model'
size_min = 5
size_max = 20

cuda_device = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
color_transform = Colorize()
transform = transforms.Compose([
    transforms.ToTensor()
])
mask_transform = transforms.Compose([
    transforms.ToTensor()
])

train_set = medicalDataLoader.MedicalImageDataset('train', root_dir, transform=transform, mask_transform=mask_transform,
                                                  augment=True, equalize=False)


def main():
    ## Here we have to split the fully annotated dataset and unannotated dataset
    split_ratio = 0.2
    random_index = np.random.permutation(len(train_set))
    labeled_dataset = copy.deepcopy(train_set)
    labeled_dataset.imgs = [train_set.imgs[x] for x in random_index[:int(len(random_index) * split_ratio)]]
    unlabeled_dataset = copy.deepcopy(train_set)
    unlabeled_dataset.imgs = [train_set.imgs[x] for x in random_index[int(len(random_index) * split_ratio):]]
    assert set(unlabeled_dataset.imgs) & set(
        labeled_dataset.imgs) == set(), \
        "there's intersection between labeled and unlabeled training set."
    
    labeled_dataLoader = DataLoader(labeled_dataset, batch_size=1, num_workers=num_workers, shuffle=True)
    unlabeled_dataLoader = DataLoader(unlabeled_dataset, batch_size=1, num_workers=num_workers, shuffle=True)
    ## Here we terminate the split of labeled and unlabeled data
    ## the validation set is for computing the dice loss.
    val_set = medicalDataLoader.MedicalImageDataset('val', root_dir, transform=transform, mask_transform=mask_transform,
                                                    equalize=False)
    val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)

    ##
    ##=====================================================================================================================#
    # np.random.choice(labeled_dataset)

    global net
    net = Enet(2)
    ## Uncomment the following line to pretrain the model with few fully labeled data.
    # pretrain(labeled_dataLoader,net,)
    map_location = lambda storage, loc: storage
    net.load_state_dict(torch.load('checkpoint/pretrained_net.pth', map_location=map_location))
    net.to(device)
    # optimiser = torch.optim.Adam(net.parameters(),lr = lr, weight_decay=1e-5)

    global labeled_img, labeled_mask, labeled_weak_mask, unlabeled_img, unlabeled_mask
    for iteration in xrange(10000):
        ## choose randomly a batch of image from labeled dataset and unlabeled dataset.
        # Initialize the ADMM dummy variables for one-batch training
        labeled_dataLoader, unlabeled_dataLoader = iter(labeled_dataLoader), iter(unlabeled_dataLoader)
        labeled_img, labeled_mask, labeled_weak_mask = next(labeled_dataLoader)[0:3]
        labeled_img, labeled_mask, labeled_weak_mask = labeled_img.to(device), labeled_mask.to(device), labeled_weak_mask.to(device)
        unlabeled_img, unlabeled_mask = next(unlabeled_dataLoader)[0:2]
        unlabeled_img, unlabeled_mask = unlabeled_img.to(device), unlabeled_mask.to(device)
        if labeled_mask.sum() == 0 or unlabeled_mask.sum() == 0:
            # skip those with no foreground masks
            continue
        f_theta_labeled = net(labeled_img)  # shape b,c,w,h
        f_theta_unlabeled = net(unlabeled_img)  # b,c,w,h
        gamma = pred2segmentation(f_theta_unlabeled).detach()  # b, w, h
        s = gamma  # b w h
        u = np.zeros(list(gamma.shape))  # b w h
        v = np.zeros(u.shape)  # b w h
        global u_r, u_s
        u_r = 1
        u_s = 1

        for i in xrange(200):
            # Finalise the initialization of ADMM dummy variable
            f_theta_labeled, f_theta_unlabeled = update_theta(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v, )
            gamma = update_gamma(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v)
            s = update_s(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v)
            u = update_u(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v)
            v = update_v(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v)

            show_image_mask(labeled_img,labeled_mask,f_theta_labeled[:,1,:,:])
            show_image_mask(unlabeled_img,unlabeled_mask,f_theta_unlabeled[:,1,:,:])
            print()


def update_theta(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v):
    global u_r, u_s, net
    global labeled_img, labeled_mask, labeled_weak_mask, unlabeled_img, unlabeled_mask
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss2d()
    for i in xrange(10):
        loss_labeled = criterion(f_theta_labeled, labeled_mask.squeeze(1))
        loss_unlabeled = u_r * (f_theta_unlabeled - gamma.float() + torch.Tensor(u).float()).norm(p=2) + \
                         u_s * (f_theta_unlabeled - s.float() + torch.Tensor(v)).norm(p=2)
        loss = loss_labeled + loss_unlabeled
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        f_theta_labeled = net(labeled_img)  # shape b,c,w,h
        f_theta_unlabeled = net(unlabeled_img)

    return f_theta_labeled, f_theta_unlabeled

import maxflow
def update_gamma(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v):
    global u_r, u_s, net
    global labeled_img, labeled_mask, labeled_weak_mask, unlabeled_img, unlabeled_mask

    unary_term_gamma_1 = np.multiply((0.5 - (f_theta_unlabeled.data.numpy()[:, 1, :, :] + u)),
                             unlabeled_img.data.numpy().squeeze(1))
    # unary_term_gamma_1 = np.zeros(unary_term_gamma_1.shape)
    unary_term_gamma_0 =np.zeros(unary_term_gamma_1.shape)
    neighbor_term = 0.1
    new_gamma = np.zeros(gamma.shape)

    for i in xrange(unary_term_gamma_1.shape[0]):
        g = maxflow.Graph[float](0, 0)
        # Add the nodes.
        nodeids = g.add_grid_nodes(list(gamma.shape)[1:])
        # Add edges with the same capacities.
        g.add_grid_edges(nodeids, neighbor_term)
        # Add the terminal edges.
        g.add_grid_tedges(nodeids, unary_term_gamma_1[i].astype(np.int32).squeeze(), unary_term_gamma_0[i].squeeze())
        g.maxflow()
        # Get the segments.
        sgm = g.get_grid_segments(nodeids)*1

        # The labels should be 1 where sgm is False and 0 otherwise.
        new_gamma[i] = np.int_(np.logical_not(sgm))
        g.reset()

    # it seems to set the unary and neighbor term of one function, one can have the result automatically.
    # In python, I failed to find the package.
    return torch.Tensor(new_gamma).float()


def update_s(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v):
    global u_r, u_s, net
    global labeled_img, labeled_mask, labeled_weak_mask, unlabeled_img, unlabeled_mask
    s_new = np.zeros(s.shape)

    a = f_theta_unlabeled.data.numpy()[:, 1, :, :] + v - 0.5
    for i in range(a.shape[0]):
        si = a[i]
        A_plus = (si >= 0).sum()
        A_sub = (si < 0).sum()
        if A_plus >= size_max:
            threshold = np.sort(si.reshape(-1))[size_max:][0]
            s_new[i][np.where(si > threshold)] = 1
        else:
            # s_new[i, :, :][np.where(si < 0)] = 1
            pass
    # return torch.Tensor(s_new).float()
    return s

def update_u(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v):
    u = u + f_theta_unlabeled.data.cpu().numpy()[:, 1:, :] - gamma.data.cpu().numpy()
    return u


def update_v(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v):
    v = v + f_theta_unlabeled.data.cpu().numpy()[:, 1:, :] - s.data.cpu().numpy()
    return v


if __name__ == "__main__":
    main()
