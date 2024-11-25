import os
import os.path as osp
from glob import glob
import pickle

import numpy as np
from PIL import Image

from lib.datasets.dataset import Dataset, Sample

TAG_FLOAT = 202021.25
TAG_CHAR = "PIEH"


def depth_read(filename):
    """Read depth data from file, return as numpy array."""
    f = open(filename, "rb")
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert (
        check == TAG_FLOAT
    ), " depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
        TAG_FLOAT, check
    )
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert (
        width > 0 and height > 0 and 1 < size < 100000000
    ), " depth_read:: Wrong input size (width = {0}, height = {1}).".format(
        width, height
    )
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def flow_read(filename):
    """Read optical flow from file, return (U,V) tuple.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    f = open(filename, "rb")
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert (
        check == TAG_FLOAT
    ), " flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
        TAG_FLOAT, check
    )
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert (
        width > 0 and height > 0 and 1 < size < 100000000
    ), " flow_read:: Wrong input size (width = {0}, height = {1}).".format(
        width, height
    )
    tmp = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width * 2))
    u = tmp[:, np.arange(width) * 2]  # H, W
    v = tmp[:, np.arange(width) * 2 + 1]  # H, W
    return u, v


def cam_read(filename):
    """Read camera data, return (M,N) tuple.

    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename, "rb")
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert (
        check == TAG_FLOAT
    ), " cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
        TAG_FLOAT, check
    )
    M = np.fromfile(f, dtype="float64", count=9).reshape((3, 3))
    N = np.fromfile(f, dtype="float64", count=12).reshape((3, 4))
    return M, N


class SintelPose:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        _, view_to_world_transform = cam_read(osp.join(root, self.path))
        return np.vstack((view_to_world_transform, [0, 0, 0, 1])).astype(np.float32)


class SintelIntrinsics:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        intrinsics, _ = cam_read(osp.join(root, self.path))
        return intrinsics


class SintelDepth:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        depth = depth_read(osp.join(root, self.path))
        depth[depth > 72] = 0.0
        return depth


class SintelFlow:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        u, v = flow_read(osp.join(root, self.path))
        flow = np.stack((u, v))  # 2, H, W
        return flow


class SintelImage:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        img_path = osp.join(root, self.path)
        img = np.array(Image.open(img_path))
        img = img.transpose([2, 0, 1])
        return img  # np array, 3xHxW, uint8


class SintelSequence:
    def __init__(self, root, name, renderpass="clean"):
        self.root = root
        self.name = name
        self.renderpass = renderpass

        image_root = osp.join(root, renderpass, name)
        frames = sorted([x[-8:-4] for x in sorted(glob(osp.join(image_root, "*.png")))])

        self.ids = frames
        self.id_to_idx = {
            int(frame): frame_idx for frame_idx, frame in enumerate(frames)
        }
        self.images = [
            SintelImage(osp.join(renderpass, name, "frame_{}.png".format(frame)))
            for frame in frames
        ]
        self.depths = [
            SintelDepth(osp.join("depth", name, "frame_{}.dpt".format(frame)))
            for frame in frames
        ]
        self.flows = [
            SintelFlow(osp.join("flow", name, "frame_{}.flo".format(frame)))
            for frame in frames
        ]
        self.poses = [
            SintelPose(osp.join("camdata_left", name, "frame_{}.cam".format(frame)))
            for frame in frames
        ]
        self.intrinsics = [
            SintelIntrinsics(
                osp.join("camdata_left", name, "frame_{}.cam".format(frame))
            )
            for frame in frames
        ]

        assert (
            len(self.images)
            == len(self.depths)
            == len(self.poses)
            == len(self.intrinsics)
        )

    def __len__(self):
        return len(self.images)


class SintelSample(Sample):

    def __init__(self, name):
        self.name = name
        self.data = {}
        self.info = {}

    def load(self, root):
        out_dict = {"_base": root, "_name": self.name}

        for key, info in self.info.items():
            out_dict["_{}".format(key)] = info

        for key, val in self.data.items():
            if not isinstance(val, list):
                if isinstance(val, np.ndarray):
                    out_dict[key] = val
                else:
                    out_dict[key] = val.load(root)
            else:
                out_dict[key] = [
                    ele if isinstance(ele, np.ndarray) else ele.load(root)
                    for ele in val
                ]

        return out_dict


class Sintel(Dataset):
    def __init__(
        self,
        sample_confs,
        type_,
        renderpass="clean",
        sequences=None,
        root=None,
        aug_fcts=None,
        to_torch=False,
        verbose=True,
    ):
        self.type_ = type_
        self.renderpass = renderpass
        root = root if root is not None else self._get_path("Sintel", "root")

        super().__init__(
            root=root,
            aug_fcts=aug_fcts,
            to_torch=to_torch,
            verbose=verbose,
            sample_confs=sample_confs,
            sequences=sequences,
        )

    def _get_sample_list_path(self):
        lists_path = self._get_path("Sintel", "lists")
        if lists_path is not None:
            return osp.join(lists_path, "{}.{}.pickle".format(self.type_, self.name))
        else:
            return None

    def _init_samples(self, sample_confs=None, sequences=None):
        sample_list_path = self._get_sample_list_path()

        if sample_list_path is not None and osp.isfile(self._get_sample_list_path()):
            self._init_samples_from_list(sample_list_path)

        elif sample_confs is not None:
            self._init_samples_from_confs(
                sample_confs=sample_confs, sequences=sequences
            )
            self.write_samples()

    def _init_samples_from_confs(self, sample_confs, sequences=None):
        image_root = osp.join(self.root, self.renderpass)

        sequence_names = sorted(os.listdir(image_root))
        sequence_names = (
            [s for s in sequence_names if s in sequences]
            if sequences is not None
            else sequence_names
        )
        sequences = [
            SintelSequence(self.root, name, self.renderpass) for name in sequence_names
        ]

        for sample_conf in sample_confs:
            for sequence in sequences:
                for frame_num in range(len(sequence)):
                    key_id = sequence.ids[frame_num]
                    key_num = int(key_id)

                    sample = SintelSample(name=sequence.name + "/key{}".format(key_id))

                    sample_valid = True
                    for key, conf in sample_conf.items():
                        if isinstance(conf, tuple):
                            data_type, offset = conf

                            offset_num = key_num + offset
                            if offset_num in sequence.id_to_idx:
                                offset_idx = sequence.id_to_idx[offset_num]
                                sample.data[key] = getattr(sequence, data_type)[
                                    offset_idx
                                ]
                            else:
                                sample_valid = False
                                break

                        elif isinstance(conf, list):
                            sample.data[key] = []

                            for conf_ in conf:
                                data_type, offset = conf_

                                offset_num = key_num + offset
                                if offset_num in sequence.id_to_idx:
                                    offset_idx = sequence.id_to_idx[offset_num]
                                    sample.data[key].append(
                                        getattr(sequence, data_type)[offset_idx]
                                    )
                                else:
                                    sample_valid = False
                                    break

                        else:
                            sample.info[key] = conf

                    if sample_valid:
                        self.samples.append(sample)

    def _init_samples_from_list(self, sample_list_path):
        if self.verbose:
            print("\tInitializing samples from list at {}.".format(sample_list_path))
        with open(sample_list_path, "rb") as sample_list:
            self.samples += pickle.load(sample_list)

    def write_samples(self, path=None):
        path = self._get_sample_list_path() if path is None else path
        super().write_samples(path)
