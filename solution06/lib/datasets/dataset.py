import abc
import os.path as osp
import pickle
import random
import time

import numpy as np
import pytoml
import torch
from torch.utils.data import Dataset as TorchDataset

from lib.utils import get_class


class Sample(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load(self, root):
        return


def _get_paths():
    paths_file = "paths.toml"
    with open(paths_file, "r") as paths_file:
        return pytoml.load(paths_file)


def _get_path(*keys):
    paths = _get_paths()
    path = None
    for idx, key in enumerate(keys):
        if key in paths:
            if (
                key in paths
                and (isinstance(paths[key], str) or isinstance(paths[key], list))
                and idx == len(keys) - 1
            ):
                path = paths[key]
            else:
                paths = paths[key]
    return path


def _val_to_torch(val):
    if isinstance(val, np.ndarray):
        if val.dtype == np.uint16:
            val = val.astype(np.int32)
        if any([s < 0 for s in val.strides]):
            val = val.copy()
        return torch.from_numpy(val).float()
    elif isinstance(val, float):
        return _val_to_torch(np.array(val, dtype=np.float32))
    elif isinstance(val, list):
        return [_val_to_torch(x) for x in val]
    else:
        return val


class Dataset(TorchDataset, metaclass=abc.ABCMeta):

    def __init__(
        self, root, name=None, aug_fcts=None, to_torch=False, verbose=True, **kwargs
    ):
        aug_fcts = [] if aug_fcts is None else aug_fcts
        aug_fcts = [aug_fcts] if not isinstance(aug_fcts, list) else aug_fcts
        self.verbose = verbose

        self._name = name
        self.root = root

        if self.verbose:
            print("Initializing dataset {} from {}.".format(self.name, self.root))

        self.init_seed = False
        self.aug_fcts = []
        self._init_aug_fcts(aug_fcts)
        self.to_torch = to_torch

        self.samples = []

        self._init_samples(**kwargs)

        if self.verbose:
            print("\tNumber of samples: {}".format(len(self)))
            print("Finished initializing dataset {}.".format(self.name))
            print()

    @property
    def name(self):
        name = self._name if self._name is not None else type(self).__name__
        return name

    def _init_aug_fcts(self, aug_fcts):
        for aug_fct in aug_fcts:
            if isinstance(aug_fct, str):
                aug_fct_class = get_class(aug_fct)
                aug_fct = aug_fct_class()
            self.aug_fcts.append(aug_fct)

    @abc.abstractmethod
    def _init_samples(self, **kwargs):
        return

    def __getitem__(self, index):
        sample = self.samples[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        sample_dict = sample.load(root=self.root)
        sample_dict["_index"] = index
        sample_dict["_dataset"] = self.name

        if "_name" not in sample_dict:
            sample_dict["_name"] = "{}/{}".format(self.name, index)

        for aug_fct in self.aug_fcts:
            aug_fct(sample_dict)

        if self.to_torch:
            for key, val in sample_dict.items():
                sample_dict[key] = _val_to_torch(val)

        return sample_dict

    def __len__(self):
        return len(self.samples)

    def write_samples(self, path):
        if osp.isdir(osp.split(path)[0]):
            print("Writing sample list to {}".format(path))
            with open(path, "wb") as file:
                pickle.dump(self.samples, file)
        else:
            print("Could not write sample list to {}".format(path))

    def _get_paths(self):
        return _get_paths()

    def _get_path(self, *keys):
        return _get_path(*keys)

    @classmethod
    def init_as_loader(
        cls,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=None,
        **kwargs
    ):
        dataset = cls(to_torch=True, **kwargs)
        return dataset.get_loader(
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
        )

    def get_loader(
        self,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=None,
    ):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
        )

    def timeit(self, num_batches=100, batch_size=1, num_workers=0):
        start = time.time()

        loader = self.get_loader(batch_size=batch_size, num_workers=num_workers)
        for idx, data_blob in enumerate(loader):
            if idx >= num_batches - 1:
                break

        end = time.time()
        print(
            "Total time for loading {} batches: %1.4fs.".format(num_batches)
            % (end - start)
        )
        print("Mean time per batch: %1.4fs." % ((end - start) / num_batches))
