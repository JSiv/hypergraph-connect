import glob
import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import canny
from torch.utils.data import Dataset
from torchvision import transforms


class CelebADataset(Dataset):

    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        super(CelebADataset, self).__init__()

        self.augment = augment
        self.training = training

        transform = transforms.Compose([
            transforms.Resize(config.INPUT_SIZE),
            transforms.CenterCrop(config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        grayscale_transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        grayscale_temp_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        temp_transform = transforms.Compose([
            transforms.Resize(config.INPUT_SIZE),
            transforms.CenterCrop(config.INPUT_SIZE),
            transforms.ToTensor()
        ])
        self.transform = transform
        self.grayscale_temp_transform = grayscale_temp_transform
        self.grayscale_transform = grayscale_transform
        self.temp_transform = temp_transform
        self.image_names = self.load_filelists(flist)
        self.masks = self.load_filelists(mask_flist)
        self.edges = self.load_filelists(edge_flist)
        self.sigma = config.SIGMA

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_path = self.image_names[index]
        print(img_path)

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        image_gray = self.grayscale_temp_transform(img)

        print(img.shape)

        print(image_gray.shape)

        mask = Image.open(img_path)
        # mask = rgb2gray(mask)
        mask = self.temp_transform(mask)

        print(mask.shape)

        # no edge
        if self.sigma == -1:
            edge = np.zeros(img.shape).astype(np.float)
        else:
            # random sigma
            if self.sigma == 0:
                self.sigma = random.randint(1, 4)

            # input = torchvision.transforms.Grayscale(img)
            edge = canny(image_gray, sigma=self.sigma, mask=mask).astype(np.float)

        image_gray = self.grayscale_transform(image_gray)

        edge = self.temp_transform(edge)
        return img, image_gray, edge, mask

    def load_filelists(self, filenames):
        if isinstance(filenames, list):
            return filenames

        # flist: image file path, image directory path, text file flist path
        if isinstance(filenames, str):
            if os.path.isdir(filenames):
                flist = list(glob.glob(filenames + '/*.jpg')) + list(glob.glob(filenames + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(filenames):
                try:
                    return np.genfromtxt(filenames, dtype=np.str, encoding='utf-8')
                except:
                    return [filenames]

        return []
