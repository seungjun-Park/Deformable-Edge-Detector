import glob
import os
import json
import random

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import instantiate_from_config, to_2tuple
from typing import Union, List, Tuple

IMG_FORMATS = ['png', 'jpg']
STR_FORMATS = ['txt', 'csv']


class BIPEDDataset(Dataset):
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
            self.img_names = glob.glob(f'{root}/edges/imgs/train/*/*/*.*')
        else:
            self.img_names = glob.glob(f'{root}/edges/imgs/test/*/*.*')

    def __getitem__(self, index):
        img_name = self.img_names[index]
        edge_name = img_name.rsplit('imgs')
        edge_name = f'{edge_name[0]}/edge_maps/{edge_name[1]}'
        edge_name = edge_name.rsplit('.', 1)[0] + '.png'

        img = cv2.imread(f'{img_name}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, self.color_space)
        edge = cv2.imread(f'{edge_name}', cv2.IMREAD_GRAYSCALE)

        img = self.to_tensor(img)
        edge = self.to_tensor(edge)
        edge = 1.0 - edge
        # edge = torch.where(edge >= 0.8, 1.0, 0.0)
        img = self.resize(img)
        edge = self.resize(edge)

        # i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=self.scale, ratio=self.ratio)
        #
        # img = tf.resized_crop(img, i, j, h, w, size=self.size, antialias=True)
        # edge = tf.resized_crop(edge, i, j, h, w, size=self.size, antialias=True)

        return img, edge

    def __len__(self):
        return len(self.img_names)
