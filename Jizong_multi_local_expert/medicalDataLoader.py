from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint

root_dir = '../ACDC-2D-All'
batch_size = 4
num_workers = 4

transform = transforms.Compose([
    transforms.ToTensor()
])
mask_transform = transforms.Compose([
    transforms.ToTensor()
])


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        train_img_path = os.path.join(root, 'train', 'Img')
        train_mask_path = os.path.join(root, 'train', 'GT')
        train_mask_weak_path = os.path.join(root, 'train', 'WeaklyAnnotations')
        
        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)
        labels_weak = os.listdir(train_mask_weak_path)

        images.sort()
        labels.sort()
        labels_weak.sort()

        for it_im, it_gt, it_w in zip(images, labels,labels_weak):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt), os.path.join(train_mask_weak_path, it_w))
            items.append(item)
            
    elif mode == 'val':
        train_img_path = os.path.join(root, 'val', 'Img')
        train_mask_path = os.path.join(root, 'val', 'GT')
        train_mask_weak_path = os.path.join(root, 'val', 'WeaklyAnnotations')
        
        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)
        labels_weak = os.listdir(train_mask_weak_path)
        
        ## 去掉第一张 没有ground truth的图像
        # images = [x for x in images if x.split('.')[0].split('_')[2]!='1' ]
        # labels = [x for x in labels if x.split('.')[0].split('_')[2]!='1']
        # labels_weak = [x for x in labels_weak if x.split('.')[0].split('_')[2]!='1']


        images.sort()
        labels.sort()
        labels_weak.sort()


         
        for it_im, it_gt, it_w in zip(images, labels,labels_weak):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt), os.path.join(train_mask_weak_path, it_w))
            items.append(item)
    else:
        train_img_path = os.path.join(root, 'test', 'Img')
        train_mask_path = os.path.join(root, 'test', 'GT')
        train_mask_weak_path = os.path.join(root, 'test', 'WeaklyAnnotations')
        
        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)
        labels_weak = os.listdir(train_mask_weak_path)
        
        images.sort()
        labels.sort()
        labels_weak.sort()
         
        for it_im, it_gt, it_w in zip(images, labels,labels_weak):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt), os.path.join(train_mask_weak_path, it_w))
            items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, transform=None, mask_transform=None, augment=False, equalize=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize

    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random() > 0.5:
            angle = random() * 90 - 45
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask

    def __getitem__(self, index):
        img_path, mask_path, mask_weak_path = self.imgs[index]
        # print("{} and {}".format(img_path,mask_path))
        img = Image.open(img_path)  # .convert('RGB')
        mask = Image.open(mask_path)  # .convert('RGB')
        mask_weak = Image.open(mask_weak_path).convert('L')
        
        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augmentation:
            img, mask = self.augment(img, mask)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)
            mask = (mask==1).long()
            # mask = self.pixelvalue2OneHot(mask)
            mask_weak = self.mask_transform(mask_weak)

        return [img, mask, mask_weak, img_path]

    def pixelvalue2OneHot(self,mask):
        possible_pixel_values = [0.000000, 0.33333334, 0.66666669, 1.000000]
        mask_ = mask.clone()
        for i,p in enumerate(possible_pixel_values):
            mask_[(mask<p+0.1) &(mask>p-0.1)]=i
        mask_ = mask_.long()
        return mask_


def mask_size_dict():
    train_set = MedicalImageDataset('train',root_dir,transform=transform,mask_transform=mask_transform,augment=False,equalize=False)
    mask_size=dict()
    for i, (img,mask,weakmask,path) in enumerate(train_set):
        mask_size[path]=mask.sum().item()
    mask_size=dict(sorted(mask_size.items(),key=lambda item:item[1]))
    return mask_size

masksizedict=mask_size_dict()


class MedicalImageDataset_LocalExpert(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, size_range,transform=None, mask_transform=None, augment=False, equalize=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)
        self.imgs = [x for x in self.imgs if (masksizedict[x[0]] >= size_range[0]) and (masksizedict[x[0]] <= size_range[1])]
        self.augmentation = augment
        self.equalize = equalize

    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random() > 0.5:
            angle = random() * 90 - 45
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask

    def __getitem__(self, index):
        img_path, mask_path, mask_weak_path = self.imgs[index]
        # print("{} and {}".format(img_path,mask_path))
        img = Image.open(img_path)  # .convert('RGB')
        mask = Image.open(mask_path)  # .convert('RGB')
        mask_weak = Image.open(mask_weak_path).convert('L')

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augmentation:
            img, mask = self.augment(img, mask)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)
            mask = (mask == 1).long()
            # mask = self.pixelvalue2OneHot(mask)
            mask_weak = self.mask_transform(mask_weak)

        return [img, mask, mask_weak, img_path]

    def pixelvalue2OneHot(self, mask):
        possible_pixel_values = [0.000000, 0.33333334, 0.66666669, 1.000000]
        mask_ = mask.clone()
        for i, p in enumerate(possible_pixel_values):
            mask_[(mask < p + 0.1) & (mask > p - 0.1)] = i
        mask_ = mask_.long()
        return mask_

if __name__=="__main__":
    train_set = MedicalImageDataset_LocalExpert('train', root_dir,(0,300) ,transform=transform, mask_transform=mask_transform,
                                    augment=False, equalize=False)
    print(len(train_set))


