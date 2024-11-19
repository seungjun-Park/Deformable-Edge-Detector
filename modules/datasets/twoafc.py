import numpy as np
import torch
import glob
import os

import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from utils import instantiate_from_config


class TwoAFCDataset(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 subdirs=['traditional'],
                 transform_configs=None,
                 ):
        if transform_configs is not None:
            transform_list = list()
            for transform_config in transform_configs['transforms']:
                transform_list.append(instantiate_from_config(transform_config))
            self.transform = transforms.Compose(transform_list)

        else:
            self.transform = None

        self.subdirs = []

        if train:
            root = os.path.join(root, 'train')
            for subdir in subdirs:
                assert subdir in ['traditional', 'cnn', 'mix'], f'{subdir} is not available.'
            self.subdirs = subdirs
        else:
            root = os.path.join(root, 'val')
            for subdir in subdirs:
                assert subdir in ['traditional', 'cnn', 'superres', 'deblur', 'color', 'frameinterp'], f'{subdir} is not available.'
            self.subdirs = subdirs

        self.root = os.path.normpath(root)
        self.file_paths = []
        for subdir in self.subdirs:
            file_path = glob.glob(f'{self.root}/{subdir}/ref/*.*')
            self.file_paths += file_path

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        root, dir_name, file_name = file_path.rsplit('/', 2)
        file_name = file_name.rsplit('.', 1)[0]

        p0_img = cv2.imread(f'{root}/p0/{file_name}.png', cv2.IMREAD_COLOR)
        p0_img = cv2.cvtColor(p0_img, cv2.COLOR_BGR2RGB)

        p1_img = cv2.imread(f'{root}/p1/{file_name}.png', cv2.IMREAD_COLOR)
        p1_img = cv2.cvtColor(p1_img, cv2.COLOR_BGR2RGB)

        ref_img = cv2.imread(f'{root}/ref/{file_name}.png', cv2.IMREAD_COLOR)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        judge_img = np.load(f'{root}/judge/{file_name}.npy').reshape((1, 1, 1, ))  # [0,1]
        judge_img = torch.from_numpy(judge_img)

        if self.transform is not None:
            p0_img = self.transform(p0_img)
            p1_img = self.transform(p1_img)
            ref_img = self.transform(ref_img)

        return ref_img, p0_img, p1_img, judge_img

    def __len__(self):
        return len(self.file_paths)
