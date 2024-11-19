import glob
import cv2
import numpy as np
import torch
import pytorch_lightning as pl
import torchvision.transforms as tf
from torch.utils.data import Dataset, DataLoader, IterableDataset

from utils import partial, instantiate_from_config

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, IterableDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, use_worker_init_fn=False,
                 shuffle_val_dataloader=False, shuffle_test_loader=False):
        super().__init__()

        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn

        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = partial(self._train_dataloader)
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._validation_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader

        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)

        if self.wrap:
            for k in self.datasets:
                self.dataset_configs[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], IterableDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = self.use_worker_init_fn
        else:
            init_fn = None

        return DataLoader(self.datasets['train'], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _validation_dataloader(self, shuffle=True):
        if isinstance(self.datasets['validation'], IterableDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=True):
        is_iterable_dataset = isinstance(self.datasets['test'], IterableDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=True):
        if isinstance(self.datasets['predict'], IterableDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def calculate_mean_std(root: str, use_color: bool = True):
    img_names = glob.glob(f'{root}/*.*')
    to_tensor = tf.ToTensor()
    means = []
    stds = []
    if use_color:
        option = cv2.IMREAD_COLOR
    else:
        option = cv2.IMREAD_GRAYSCALE

    for name in img_names:
        img = cv2.imread(f'{name}', option)
        if use_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = to_tensor(img)
        means.append(torch.mean(img, dim=[-2, -1]))
        stds.append(torch.std(img, dim=[-2, -1]))

    if use_color:
        mean_r = np.mean([m[0] for m in means])
        mean_g = np.mean([m[1] for m in means])
        mean_b = np.mean([m[2] for m in means])

        std_r = np.mean([s[0] for s in stds])
        std_g = np.mean([s[1] for s in stds])
        std_b = np.mean([s[2] for s in stds])

        mean = [mean_r, mean_g, mean_b]
        std = [std_r, std_g, std_b]

    else:
        mean = np.mean([m[0] for m in means])
        std = np.mean([s[0] for s in stds])

    return mean, std