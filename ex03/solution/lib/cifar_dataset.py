"""CIFAR10 dataset."""

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import pickle

import os
import shutil
import sys
import json
from typing import Dict, List, Optional, Any, Iterable, Mapping, Tuple, Union, Callable, BinaryIO
from pathlib import Path

import torch as th
from torch import nn
from pprint import pprint
import time
from timeit import default_timer as timer
from enum import Enum


def create_cifar_datasets(path: str = "./data", transform_custom=transforms.ToTensor()
                          ) -> Tuple[Dataset, Dataset]:
    """
    Setup CIFAR10 train and test set.

    Args:
        path: Target path to store the downloaded data
        transform_custom: Transformations to apply to the data

    Returns:
        Tuple of train and test dataset
    """
    train_set = datasets.CIFAR10(path, train=True, download=True, transform=transform_custom)
    test_set = datasets.CIFAR10(path, train=False, download=True, transform=transform_custom)

    return train_set, test_set


def create_dataloader(dataset: Dataset, batch_size: int, is_train: bool = True,
                      num_workers: int = 0) -> DataLoader:
    """
    Given a dataset, create the dataloader.

    Args:
        dataset: Input dataset
        batch_size: Batch size
        is_train: Whether this is a dataloader for training or for testing
        num_workers: How many processes to use for dataloading

    Returns:
        dataloader
    """
    # Create an instance of the DataLoader class given the dataset, batch_size and num_workers.
    # Set the shuffle parameter to True if is_train is True, otherwise set it to False.
    return DataLoader(dataset, batch_size, shuffle=is_train, num_workers=num_workers)


def get_dataloaders(args, train_transforms, val_transforms):
    # define dataset
    datafolder = os.path.join(args.out_dir, '../data')
    trainset = CIFAR10_custom(root=datafolder, subsample_factor=args.subsample_factor,
                              download=True, transform=train_transforms,
                              split='train')
    valset = CIFAR10_custom(root=datafolder, subsample_factor=args.subsample_factor,
                            download=True, transform=val_transforms,
                            split='val')
    testset = CIFAR10_custom(root=datafolder, subsample_factor=args.subsample_factor,
                             download=True, transform=val_transforms,
                             split='test')
    # get dataloader
    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    # get dataloader
    val_loader = DataLoader(valset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    # get test_loader
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


class CIFAR10_custom(datasets.CIFAR10):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        split (str, optional): Which split to use (train, val or test). Default: train
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        val_split: Size of validation set.
        subsample_factor: Factor to subsample the dataset by.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
            ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
            ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
            ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
            'filename': 'batches.meta',
            'key': 'label_names',
            'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            val_split: float = 0.1,
            subsample_factor: float = 1.0
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform, download=download)

        self.train = split == 'train' or split == 'val'  # training set or test set

        if split == 'train' or split == 'val':
            downloaded_list = self.train_list
        elif split == 'test':
            downloaded_list = self.test_list
        else:
            raise Exception('unkown data split')

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape((-1, 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        split_idx = int(len(self.data) * (1 - val_split))
        if split == 'train':
            self.data = self.data[:split_idx]
            self.targets = self.targets[:split_idx]
        elif split == 'val':
            self.data = self.data[split_idx:]
            self.targets = self.targets[split_idx:]

        # reduce dataset size for faster overfitting
        self.data = self.data[:int(len(self.data) * subsample_factor)]
        self.targets = self.targets[:len(self.data)]

        self._load_meta()
