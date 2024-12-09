import os
import os.path as osp
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
        use_subset=False,
        root=None,
        aug_fcts=None,
        to_torch=False,
        verbose=True,
    ):
        self.split = split
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
            use_subset=use_subset,
        )

    def _init_samples(self, sample_confs, use_subset=False):
        filtered_samples = self.get_subset_filtered_frames() if use_subset else []
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

                            exclude_file = False
                            for filtered_sample in filtered_samples:
                                if (
                                    filtered_sample[0] in sequence
                                    and filtered_sample[1] in file
                                ):
                                    exclude_file = True

                            if not exclude_file:
                                files[key][frame_num] = osp.join(conf.path, file)
                                frame_nums.add(frame_num)

                    elif isinstance(conf, list):
                        files[key] = []

                        for conf_idx, conf_ in enumerate(conf):
                            files[key].append({})

                            for file in os.listdir(osp.join(sequence, conf_.path)):
                                frame_num = int(osp.splitext(file)[0])

                                exclude_file = False

                                for filtered_sample in filtered_samples:
                                    if (
                                        filtered_sample[0] in sequence
                                        and filtered_sample[1] in file
                                    ):
                                        exclude_file = True

                                if not exclude_file:
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

    def get_subset_filtered_frames(self):
        subset_filtered_frames_path = osp.join(self.root, "subset_filtered_frames.txt")
        with open(subset_filtered_frames_path, "r") as subset_filtered_frames_file:
            subset_filtered_frames = subset_filtered_frames_file.readlines()
            subset_filtered_frames = [x[:-1] for x in subset_filtered_frames]
            subset_filtered_frames = [x.split(":") for x in subset_filtered_frames]

        return subset_filtered_frames
