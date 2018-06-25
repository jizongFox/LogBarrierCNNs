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

board_image = Dashboard(server='http://turing.livia.etsmtl.ca', env="ADMM_image")
board_loss = Dashboard(server='http://turing.livia.etsmtl.ca', env="ADMM_loss")

use_gpu = True
device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

batch_size = 1
batch_size_val = 1
num_workers = 0
lr = 0.001
max_epoch = 100
root_dir = '../ACDC-2D-All'
model_dir = 'model'
size_min = 5
size_max = 2000

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
        labeled_dataset.imgs) == set(), "there's intersection between labeled and unlabeled training set."
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
        unlabeled_img, unlabeled_mask = next(unlabeled_dataLoader)[0:2]
        if labeled_mask.sum() == 0 or unlabeled_mask.sum() == 0:
            # skip those with no foreground masks
            continue
        f_theta_labeled = net(labeled_img)  # shape b,c,w,h
        f_theta_unlabeled = net(unlabeled_img)  # b,c,w,h
        gamma = pred2segmentation(f_theta_unlabeled).detach()  # b, w, h
        s = gamma  # b w h
        u = np.random.randn(*list(gamma.shape))  # b w h
        v = np.random.randn(*u.shape)  # b w h
        global u_r, u_s
        u_r = 1
        u_s = 1
        # Finalise the initialization of ADMM dummy variable
        f_theta_labeled, f_theta_unlabeled = update_theta(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v, )
        gamma = update_gamma(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v)
        s = update_s(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v)


def update_theta(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v):
    global u_r, u_s, net
    global labeled_img, labeled_mask, labeled_weak_mask, unlabeled_img, unlabeled_mask
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss2d()
    for i in xrange(10):
        loss_labeled = criterion(f_theta_labeled, labeled_mask.squeeze(1))
        loss_unlabeled = u_r * (f_theta_unlabeled - gamma.float() + torch.Tensor(u)).norm(p=2) + \
                         u_s * (f_theta_unlabeled - s.float() + torch.Tensor(v)).norm(p=2)
        loss = loss_labeled + loss_unlabeled
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        f_theta_labeled = net(labeled_img)  # shape b,c,w,h
        f_theta_unlabeled = net(unlabeled_img)

    return f_theta_labeled, f_theta_unlabeled


def update_gamma(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v):
    global u_r, u_s, net
    global labeled_img, labeled_mask, labeled_weak_mask, unlabeled_img, unlabeled_mask

    unary_term = np.multiply((0.5 - (f_theta_unlabeled.data.numpy()[:, 1, :, :] + u)),
                             unlabeled_img.data.numpy().squeeze(1))
    neighbor_term = None

    # it seems to set the unary and neighbor term of one function, one can have the result automatically.
    # In python, I failed to find the package.
    return gamma


def update_s(f_theta_labeled, f_theta_unlabeled, gamma, s, u, v):
    global u_r, u_s, net
    global labeled_img, labeled_mask, labeled_weak_mask, unlabeled_img, unlabeled_mask
    s_new = np.zeros(s.shape)

    a = f_theta_unlabeled.data.numpy()[:, 1, :, :] + v - 0.5
    for i in range(a.shape[0]):
        si = a[i]
        A_plus = (si >= 0).sum()
        A_sub = (si < 0).sum()

        if si.sum() >= size_max:
            pass
        else:
            pass


if __name__ == "__main__":
    main()
