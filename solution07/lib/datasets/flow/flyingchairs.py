import os.path as osp
from glob import glob

import numpy as np

from lib.datasets.dataset import Dataset, Sample
from lib.datasets.IO import read


class FlyingChairsSample(Sample):

    def __init__(self, name, images, flow):
        self.name = name
        self.data = {"images": images, "gt_flow": flow}
        self.info = {}

    def load(self, root):
        out_dict = {"_base": root, "_name": self.name}

        for key, info in self.info.items():
            out_dict["_{}".format(key)] = info

        for key, val in self.data.items():
            if isinstance(val, list):
                out_dict[key] = [read(osp.join(root, file)) for file in val]
            else:
                out_dict[key] = read(osp.join(root, val))

        return out_dict


class FlyingChairs(Dataset):
    def __init__(
        self, split=None, root=None, aug_fcts=None, to_torch=False, verbose=True
    ):
        root = root if root is not None else self._get_path("FlyingChairs", "root")

        super().__init__(
            root=root,
            aug_fcts=aug_fcts,
            to_torch=to_torch,
            verbose=verbose,
            split=split,
        )

    def _init_samples(self, split=None):
        # adapted from https://github.com/princeton-vl/RAFT/blob/master/core/datasets.py
        images = [osp.split(p)[1] for p in sorted(glob(osp.join(self.root, "*.ppm")))]
        flows = [osp.split(p)[1] for p in sorted(glob(osp.join(self.root, "*.flo")))]
        assert len(images) // 2 == len(flows)

        split_list = np.loadtxt(
            osp.join(self.root, "FlyingChairs_train_val.txt"), dtype=np.int32
        )
        for i in range(len(flows)):
            xid = split_list[i]
            if (
                (split is None)
                or (split == "training" and xid == 1)
                or (split == "validation" and xid == 2)
            ):
                sample = FlyingChairsSample(
                    name=flows[i][:5],
                    images=[images[2 * i], images[2 * i + 1]],
                    flow=flows[i],
                )
                self.samples.append(sample)


class FlyingChairsTrain(FlyingChairs):

    def __init__(self, root=None, aug_fcts=None, to_torch=False, verbose=True):
        split = "training"
        super().__init__(
            split=split,
            root=root,
            aug_fcts=aug_fcts,
            to_torch=to_torch,
            verbose=verbose,
        )


class FlyingChairsTest(FlyingChairs):

    def __init__(self, root=None, aug_fcts=None, to_torch=False, verbose=True):
        split = "validation"
        super().__init__(
            split=split,
            root=root,
            aug_fcts=aug_fcts,
            to_torch=to_torch,
            verbose=verbose,
        )
