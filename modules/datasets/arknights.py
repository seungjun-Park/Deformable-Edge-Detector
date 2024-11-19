import glob
import os
import json
import random

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tf
from typing import Union, List, Tuple

from utils import instantiate_from_config, to_2tuple

IMG_FORMATS = ['png', 'jpg']
STR_FORMATS = ['txt', 'csv']


class ArknightsDataset(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 size: Union[int, List[int], Tuple[int]] = 224,
                 scale: Union[List[float], Tuple[float]] = (0.08, 1.0),
                 ratio: Union[List[float], Tuple[float]] = (0.75, 1.3333333333333333),
                 color_space: str = 'rgb',
                 ):
        super().__init__()
        color_space = color_space.lower()
        if color_space == 'rgb':
            self.color_space = cv2.COLOR_BGR2RGB
        elif color_space == 'rgba':
            self.color_space = cv2.COLOR_BGR2RGBA
        elif color_space == 'gray':
            self.color_space = cv2.COLOR_BGR2GRAY
        elif color_space == 'xyz':
            self.color_space = cv2.COLOR_BGR2XYZ
        elif color_space == 'ycrcb':
            self.color_space = cv2.COLOR_BGR2YCrCb
        elif color_space == 'hsv':
            self.color_space = cv2.COLOR_BGR2HSV
        elif color_space == 'lab':
            self.color_space = cv2.COLOR_BGR2LAB
        elif color_space == 'luv':
            self.color_space = cv2.COLOR_BGR2LUV
        elif color_space == 'hls':
            self.color_space = cv2.COLOR_BGR2HLS
        elif color_space == 'yuv':
            self.color_space = cv2.COLOR_BGR2YUV

        self.to_tensor = transforms.ToTensor()

        self.size = list(to_2tuple(size))
        self.scale = list(to_2tuple(scale))
        self.ratio = list(to_2tuple(ratio))

        self.resize = transforms.Resize(size=self.size, antialias=True)

        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')

        self.edge_names = glob.glob(f'{root}/*/edges/*.*')
        self.img_names = glob.glob(f'{root}/*/images/*.*')

        self.color_jitter = transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0.5, hue=0.5)
        self.invert = transforms.RandomInvert(p=1.0)
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)

        assert len(self.edge_names) == len(self.img_names)

    def __getitem__(self, index):
        edge_name = self.edge_names[index]
        img_name = self.img_names[index]

        img = cv2.imread(f'{img_name}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, self.color_space)
        edge = cv2.imread(f'{edge_name}', cv2.IMREAD_GRAYSCALE)

        img = self.to_tensor(img)
        edge = self.to_tensor(edge)

        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=self.scale, ratio=self.ratio)

        img = tf.resized_crop(img, i, j, h, w, size=self.size, antialias=True)
        edge = tf.resized_crop(edge, i, j, h, w, size=self.size, antialias=True)

        if random.random() < 0.5:
            if random.random() < 0.5:
                img = self.color_jitter(img)
            else:
                img = self.invert(img)

        if random.random() < 0.5:
            img = self.horizontal_flip(img)
            edge = self.horizontal_flip(edge)

        return img, edge

    def __len__(self):
        return len(self.img_names)


class ArknightsTripletDataset(Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 size: Union[int, List[int], Tuple[int]] = 224,
                 scale: Union[List[float], Tuple[float]] = (0.08, 1.0),
                 ratio: Union[List[float], Tuple[float]] = (0.75, 1.3333333333333333),
                 color_space: str = 'rgb',
                 ):
        super().__init__()

        color_space = color_space.lower()
        if color_space == 'rgb':
            self.color_space = cv2.COLOR_BGR2RGB
        elif color_space == 'rgba':
            self.color_space = cv2.COLOR_BGR2RGBA
        elif color_space == 'gray':
            self.color_space = cv2.COLOR_BGR2GRAY
        elif color_space == 'xyz':
            self.color_space = cv2.COLOR_BGR2XYZ
        elif color_space == 'ycrcb':
            self.color_space = cv2.COLOR_BGR2YCrCb
        elif color_space == 'hsv':
            self.color_space = cv2.COLOR_BGR2HSV
        elif color_space == 'lab':
            self.color_space = cv2.COLOR_BGR2LAB
        elif color_space == 'luv':
            self.color_space = cv2.COLOR_BGR2LUV
        elif color_space == 'hls':
            self.color_space = cv2.COLOR_BGR2HLS
        elif color_space == 'yuv':
            self.color_space = cv2.COLOR_BGR2YUV

        self.to_tensor = transforms.ToTensor()

        self.size = list(to_2tuple(size))
        self.scale = list(to_2tuple(scale))
        self.ratio = list(to_2tuple(ratio))

        self.resize = transforms.Resize(self.size)

        if train:
            root = os.path.join(root, 'train')

        else:
            root = os.path.join(root, 'val')

        self.edge_names = glob.glob(f'{root}/*/edges/*.*')
        self.img_names = glob.glob(f'{root}/*/images/*.*')

        assert len(self.edge_names) == len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        edge_pos_name = self.edge_names[index]
        while True:
            edge_neg_idx = random.randrange(0, len(self))
            if edge_neg_idx != index:
                break
        edge_neg_name = self.edge_names[edge_neg_idx]

        img = cv2.imread(f'{img_name}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        edge_pos = cv2.imread(f'{edge_pos_name}', cv2.IMREAD_GRAYSCALE)
        edge_neg = cv2.imread(f'{edge_neg_name}', cv2.IMREAD_GRAYSCALE)

        img = self.to_tensor(img)
        edge_pos = self.to_tensor(edge_pos)
        edge_neg = self.to_tensor(edge_neg)

        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=self.scale, ratio=self.ratio)

        img = tf.resized_crop(img, i, j, h, w, size=self.size, antialias=True)
        edge_pos = tf.resized_crop(edge_pos, i, j, h, w, size=self.size, antialias=True)
        edge_neg = tf.resized_crop(edge_neg, i, j, h, w, size=self.size, antialias=True)

        edge_pos = edge_pos.repeat(3, 1, 1)
        edge_neg = edge_neg.repeat(3, 1, 1)

        return img, edge_pos, edge_neg

    def __len__(self):
        return len(self.img_names)


class ArknightsImageEdgeSimilarity(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 size: Union[int, List[int], Tuple[int]] = 224,
                 scale: Union[List[float], Tuple[float]] = (0.08, 1.0),
                 ratio: Union[List[float], Tuple[float]] = (0.75, 1.3333333333333333),
                 color_space: str = 'rgb',
                 ):
        super().__init__()
        color_space = color_space.lower()
        if color_space == 'rgb':
            self.color_space = cv2.COLOR_BGR2RGB
        elif color_space == 'rgba':
            self.color_space = cv2.COLOR_BGR2RGBA
        elif color_space == 'gray':
            self.color_space = cv2.COLOR_BGR2GRAY
        elif color_space == 'xyz':
            self.color_space = cv2.COLOR_BGR2XYZ
        elif color_space == 'ycrcb':
            self.color_space = cv2.COLOR_BGR2YCrCb
        elif color_space == 'hsv':
            self.color_space = cv2.COLOR_BGR2HSV
        elif color_space == 'lab':
            self.color_space = cv2.COLOR_BGR2LAB
        elif color_space == 'luv':
            self.color_space = cv2.COLOR_BGR2LUV
        elif color_space == 'hls':
            self.color_space = cv2.COLOR_BGR2HLS
        elif color_space == 'yuv':
            self.color_space = cv2.COLOR_BGR2YUV

        self.to_tensor = transforms.ToTensor()

        self.size = list(to_2tuple(size))
        self.scale = list(to_2tuple(scale))
        self.ratio = list(to_2tuple(ratio))

        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')

        self.edge_names = glob.glob(f'{root}/*/edges/*.*')
        self.img_names = glob.glob(f'{root}/*/images/*.*')

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = cv2.imread(f'{img_name}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, self.color_space)
        img = self.to_tensor(img)

        if random.random() > 0.5:
            edge_name = self.edge_names[index]
            edge = cv2.imread(f'{edge_name}', cv2.IMREAD_COLOR)
            edge = cv2.cvtColor(edge, self.color_space)
            edge = self.to_tensor(edge)
            label = torch.Tensor([0.0])

        else:
            idx = random.randrange(0, len(self))
            while True:
                if idx != index:
                    break
                idx = random.randrange(0, len(self))

            edge_name = self.edge_names[idx]
            edge = cv2.imread(f'{edge_name}', cv2.IMREAD_COLOR)
            edge = cv2.cvtColor(edge, self.color_space)
            edge = self.to_tensor(edge)
            label = torch.Tensor([1.0])

        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=self.scale, ratio=self.ratio)
        img = tf.resized_crop(img, i, j, h, w, size=self.size, antialias=True)
        edge = tf.resized_crop(edge, i, j, h, w, size=self.size, antialias=True)

        return img, edge, label

    def __len__(self):
        return len(self.img_names)


class ArknightsImageEdgeClassification(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 size: Union[int, List[int], Tuple[int]] = 224,
                 scale: Union[List[float], Tuple[float]] = (0.08, 1.0),
                 ratio: Union[List[float], Tuple[float]] = (0.75, 1.3333333333333333),
                 color_space: str = 'rgb',
                 ):
        super().__init__()
        color_space = color_space.lower()
        if color_space == 'rgb':
            self.color_space = cv2.COLOR_BGR2RGB
        elif color_space == 'rgba':
            self.color_space = cv2.COLOR_BGR2RGBA
        elif color_space == 'gray':
            self.color_space = cv2.COLOR_BGR2GRAY
        elif color_space == 'xyz':
            self.color_space = cv2.COLOR_BGR2XYZ
        elif color_space == 'ycrcb':
            self.color_space = cv2.COLOR_BGR2YCrCb
        elif color_space == 'hsv':
            self.color_space = cv2.COLOR_BGR2HSV
        elif color_space == 'lab':
            self.color_space = cv2.COLOR_BGR2LAB
        elif color_space == 'luv':
            self.color_space = cv2.COLOR_BGR2LUV
        elif color_space == 'hls':
            self.color_space = cv2.COLOR_BGR2HLS
        elif color_space == 'yuv':
            self.color_space = cv2.COLOR_BGR2YUV

        self.to_tensor = transforms.ToTensor()

        self.size = list(to_2tuple(size))
        self.scale = list(to_2tuple(scale))
        self.ratio = list(to_2tuple(ratio))

        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')

        self.edge_names = glob.glob(f'{root}/*/edges/*.*')
        self.img_names = glob.glob(f'{root}/*/images/*.*')

    def __getitem__(self, index):
        if random.random() > 0.5:
            name = self.img_names[index]
            label = torch.Tensor([1.0])
            if random.random() > 0.5:
                x = cv2.imread(f'{name}', cv2.IMREAD_COLOR)
                x = cv2.cvtColor(x, self.color_space)
            else:
                x = cv2.imread(f'{name}', cv2.IMREAD_GRAYSCALE)
        else:
            name = self.edge_names[index]
            label = torch.Tensor([0.0])
            x = cv2.imread(f'{name}', cv2.IMREAD_GRAYSCALE)
        x = self.to_tensor(x)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)

        i, j, h, w = transforms.RandomResizedCrop.get_params(x, scale=self.scale, ratio=self.ratio)
        x = tf.resized_crop(x, i, j, h, w, size=self.size, antialias=True)

        return x, label

    def __len__(self):
        return len(self.img_names)


class ArknightsEdge(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 size: Union[int, List[int], Tuple[int]] = 224,
                 scale: Union[List[float], Tuple[float]] = (0.08, 1.0),
                 ratio: Union[List[float], Tuple[float]] = (0.75, 1.3333333333333333),
                 color_space: str = 'rgb',
                 ):
        super().__init__()
        color_space = color_space.lower()
        if color_space == 'rgb':
            self.color_space = cv2.COLOR_BGR2RGB
        elif color_space == 'rgba':
            self.color_space = cv2.COLOR_BGR2RGBA
        elif color_space == 'gray':
            self.color_space = cv2.COLOR_BGR2GRAY
        elif color_space == 'xyz':
            self.color_space = cv2.COLOR_BGR2XYZ
        elif color_space == 'ycrcb':
            self.color_space = cv2.COLOR_BGR2YCrCb
        elif color_space == 'hsv':
            self.color_space = cv2.COLOR_BGR2HSV
        elif color_space == 'lab':
            self.color_space = cv2.COLOR_BGR2LAB
        elif color_space == 'luv':
            self.color_space = cv2.COLOR_BGR2LUV
        elif color_space == 'hls':
            self.color_space = cv2.COLOR_BGR2HLS
        elif color_space == 'yuv':
            self.color_space = cv2.COLOR_BGR2YUV

        self.to_tensor = transforms.ToTensor()

        self.size = list(to_2tuple(size))
        self.scale = list(to_2tuple(scale))
        self.ratio = list(to_2tuple(ratio))

        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')

        self.edge_names = glob.glob(f'{root}/*/edges/*.*')

    def __getitem__(self, index):
        name = self.edge_names[index]
        edge = cv2.imread(f'{name}', cv2.IMREAD_COLOR)
        edge = cv2.cvtColor(edge, self.color_space)
        edge = self.to_tensor(edge)
        i, j, h, w = transforms.RandomResizedCrop.get_params(edge, scale=self.scale, ratio=self.ratio)
        edge = tf.resized_crop(edge, i, j, h, w, size=self.size, antialias=True)

        return edge, torch.tensor([0.0])

    def __len__(self):
        return len(self.edge_names)
