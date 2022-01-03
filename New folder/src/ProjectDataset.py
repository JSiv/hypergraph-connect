import glob
import os.path

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

from torchvision.transforms import transforms, InterpolationMode


class ProjectDataset(Dataset):

    def __init__(self, images, masks, input_size=256, sigma=2., augment=True, train=False):
        super(ProjectDataset, self).__init__()

        self.train = False
        self.augment = augment
        self.images = self.load_flist(dir=images)
        self.masks = self.load_flist(dir=masks)
        self.input_size = input_size
        self.sigma = sigma
        self.train = train
        # self.edges =

        self.no_of_images = len(self.images)
        self.no_of_masks = len(self.masks)

        self.transformed_images = self.transform_images(input_size)
        self.transformed_masks = self.transform_masks(input_size)

    def __getitem__(self, item):

        # Load Images
        image = Image.open(self.images[item % self.no_of_images])
        image = self.transformed_images(image.convert('RGB'))

        # Load Masks
        if self.train:
            mask = Image.open(self.masks[random.randint(0, self.no_of_masks - 1)])
        else:
            mask = Image.open(self.masks[item % self.no_of_masks])
        mask = self.transformed_masks(mask)

        threshold = 0.5
        ones = mask >= threshold
        zeros = mask < threshold

        mask.masked_fill_(ones, 1.0)
        mask.masked_fill_(zeros, 0.0)

        mask = 1 - mask

        edge, gray_image = self.image_to_edge(image, sigma=self.sigma)

        return image, gray_image, edge, mask

    def __len__(self):
        return self.no_of_images

    def load_name(self, index):
        return os.path.basename(self.images[index])

    def transform_images(self, input_size):
        return transforms.Compose([
            # transforms.CenterCrop(size=(178, 178)),  # for CelebA
            transforms.Resize(size=input_size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def transform_masks(self, input_size):
        return transforms.Compose([
            transforms.Resize(size=input_size, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def tensor_to_image(self):

        return transforms.ToPILImage()

    def image_to_edge(self, image, sigma):

        temp = rgb2gray(np.array(self.tensor_to_image()(image)))

        gray = transforms.ToTensor()(
            Image.fromarray(temp)
        )

        edge = transforms.ToTensor()(
            Image.fromarray(canny(temp, sigma))
        )
        return edge, gray

    def load_flist(self, dir):
        if isinstance(dir, list):
            return dir

        # flist: image file path, image directory path, text file flist path
        if isinstance(dir, str):
            if os.path.isdir(dir):
                flist = list(glob.glob(dir + '/*.jpg')) + list(glob.glob(dir + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(dir):
                try:
                    return np.genfromtxt(dir, dtype=np.str, encoding='utf-8')
                except:
                    return [dir]

        return []

    def load_files(self, dir, max_dataset_size=float("inf")):
        files = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        ext = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.TIF',
               '.tiff', '.TIFF', ]
        for root, _, fnames in sorted(os.walk(dir)):
            for name in fnames:
                if any(name.endswith(extension) for extension in ext):
                    path = os.path.join(root, name)
                    files.append(path)

        files = sorted(files)
        return files[:min(max_dataset_size, len(files))]


def create_dataset(opts):
    # opts.input_size
    return ProjectDataset(
        opts.image,
        opts.mask,
        256,
        opts.sigma,
        opts.mode == 'train'
    )
