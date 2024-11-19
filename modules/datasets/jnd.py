import numpy as np
import torch
import glob
import os

import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from utils import instantiate_from_config


class JNDDataset(Dataset):
    def __init__(self,
                 root,
                 dataset_type='cnn',
                 train=True,
                 transform_configs=None,
                 ):
        if transform_configs is not None:
            transform_list = list()
            for transform_config in transform_configs['transforms']:
                transform_list.append(instantiate_from_config(transform_config))
            self.transform = transforms.Compose(transform_list)

        else:
            self.transform = None

        if train:
            root = os.path.join(root, 'train', dataset_type)

        else:
            root = os.path.join(root, 'test', dataset_type)

        self.root = root

        file_path = glob.glob(f'{self.root}/ref/*.*')
        file_names = list()
        for path in file_path:
            file_name, file_format = path.rsplit('.', 1)
            file_name = file_name.rsplit('/', 1)[-1]
            if file_name in file_names:
                continue
            file_names.append(file_name)

        self.file_names = list(set(file_names))

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        p0_img = cv2.imread(f'{self.root}/p0/{file_name}.png', cv2.IMREAD_COLOR)
        p0_img = cv2.cvtColor(p0_img, cv2.COLOR_BGR2RGB)

        p1_img = cv2.imread(f'{self.root}/p1/{file_name}.png', cv2.IMREAD_COLOR)
        p1_img = cv2.cvtColor(p1_img, cv2.COLOR_BGR2RGB)

        ref_img = cv2.imread(f'{self.root}/ref/{file_name}.png', cv2.IMREAD_COLOR)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        judge_img = np.load(f'{self.root}/same/{file_name}.npy').reshape((1, 1, 1,))  # [0,1]
        judge_img = torch.FloatTensor(judge_img)

        if self.transform is not None:
            p0_img = self.transform(p0_img)
            p1_img = self.transform(p1_img)
            ref_img = self.transform(ref_img)

        return ref_img, p0_img, p1_img, judge_img

    def __len__(self):
        return len(self.file_names)