import os
import os.path as osp
import pickle
from glob import glob
from pathlib import Path

from lib.datasets.IO import read
from lib.datasets.dataset import Dataset, Sample


class DataConf:
    def __init__(self, id_, perspective=None, offset=0):
        self.id_ = id_
        self.perspective = perspective
        self.offset = offset

    @property
    def path(self):
        if self.perspective is None:
            return self.id_
        else:
            return osp.join(self.id_, self.perspective)


class FlyingThings3DSample(Sample):

    def __init__(self, base, name):
        self.base = base
        self.name = name
        self.data = {}
        self.info = {}

    def load(self, root):
        base = osp.join(root, self.base)
        out_dict = {"_base": base, "_name": self.name}

        for key, info in self.info.items():
            out_dict["_{}".format(key)] = info

        for key, val in self.data.items():
            if isinstance(val, list):
                out_dict[key] = [read(osp.join(base, file)) for file in val]
            else:
                out_dict[key] = read(osp.join(base, val))

        return out_dict


class FlyingThings3D(Dataset):
    def __init__(
        self,
        sample_confs,
        split,
        typ,
        root=None,
        aug_fcts=None,
        to_torch=False,
        verbose=True,
    ):
        self.split = split
        self.typ = typ
        root = (
            root
            if root is not None
            else self._get_path("FlyingThings3D", self.split, "root")
        )

        super().__init__(
            root=root,
            aug_fcts=aug_fcts,
            to_torch=to_torch,
            verbose=verbose,
            sample_confs=sample_confs,
        )

    def _get_sample_list_path(self):
        lists_path = self._get_path("FlyingThings3D", self.split, "lists")
        if lists_path is not None:
            return osp.join(lists_path, "{}.{}.pickle".format(self.typ, self.name))
        else:
            return None

    def _init_samples(self, sample_confs=None):
        sample_list_path = self._get_sample_list_path()

        if sample_list_path is not None and osp.isfile(self._get_sample_list_path()):
            self._init_samples_from_list(sample_list_path)

        elif sample_confs is not None:
            self._init_samples_from_confs(sample_confs)
            self.write_samples()

    def _init_samples_from_list(self, sample_list_path):
        if self.verbose:
            print("\tInitializing samples from list at {}.".format(sample_list_path))
        with open(sample_list_path, "rb") as sample_list:
            self.samples += pickle.load(sample_list)

    def _init_samples_from_confs(self, sample_confs):
        sequences = sorted(glob(osp.join(self.root, "*/*[0-9]")))
        sequences = [Path(s).as_posix() for s in sequences]

        for sample_conf in sample_confs:
            for sequence in sequences:
                files = {}
                frame_nums = set()

                for key, conf in sample_conf.items():
                    if not (isinstance(conf, DataConf) or isinstance(conf, list)):
                        continue

                    if isinstance(conf, DataConf):
                        files[key] = {}

                        for file in os.listdir(osp.join(sequence, conf.path)):
                            frame_num = int(osp.splitext(file)[0])
                            files[key][frame_num] = osp.join(conf.path, file)
                            frame_nums.add(frame_num)

                    elif isinstance(conf, list):
                        files[key] = []

                        for conf_idx, conf_ in enumerate(conf):
                            files[key].append({})

                            for file in os.listdir(osp.join(sequence, conf_.path)):
                                frame_num = int(osp.splitext(file)[0])
                                files[key][conf_idx][frame_num] = osp.join(
                                    conf_.path, file
                                )
                                frame_nums.add(frame_num)

                for frame_num in frame_nums:
                    sample = FlyingThings3DSample(
                        base=osp.relpath(sequence, self.root),
                        name=osp.relpath(sequence, self.root)
                        + "/key{:02d}".format(frame_num),
                    )

                    sample_valid = True
                    for key, conf in sample_conf.items():
                        if not (isinstance(conf, DataConf) or isinstance(conf, list)):
                            sample.info[key] = conf
                            continue

                        if isinstance(conf, DataConf):
                            offset_num = frame_num + conf.offset
                            if offset_num in files[key]:
                                sample.data[key] = files[key][offset_num]
                            else:
                                sample_valid = False
                                break

                        elif isinstance(conf, list):
                            sample.data[key] = []

                            for conf_idx, conf_ in enumerate(conf):
                                offset_num = frame_num + conf_.offset
                                if offset_num in files[key][conf_idx]:
                                    sample.data[key].append(
                                        files[key][conf_idx][offset_num]
                                    )
                                else:
                                    sample_valid = False
                                    break

                    if sample_valid:
                        self.samples.append(sample)

    def write_samples(self, path=None):
        path = self._get_sample_list_path() if path is None else path
        super().write_samples(path)
