import importlib
import os
import os.path as osp
from pathlib import Path

import pytoml
import torch
import torch.nn as nn


def sample_to_device(data, device):
    if isinstance(data, dict):
        return {key: sample_to_device(data[key], device) for key in data.keys()}
    elif isinstance(data, (list, tuple)):
        return [sample_to_device(val, device) for val in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device=device)
    else:
        return data


def get_checkpoint(auto_restore: str):
    restore_map = pytoml.load(Path("models.toml").open("r", encoding="utf-8"))
    basepaths = restore_map.pop("basepaths")
    available_models = [
        f"{model_class}/{model_name}"
        for model_class, models in restore_map.items()
        for model_name, ckpt_dict in models.items()
    ]
    try:
        model_class, model_name = auto_restore.split("/")
        ckpt_path = restore_map[model_class][model_name]["ckpt"]
    except Exception as e:
        raise ValueError(
            f"Could not find model '{auto_restore}' in models.toml. "
            f"Available models: {available_models}"
        ) from e
    ckpt_path_out = osp.join(basepaths[model_class], ckpt_path)
    return ckpt_path_out


def get_function(name):
    """from https://github.com/aschampion/diluvian/blob/master/diluvian/util.py"""
    mod_name, func_name = name.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func


def get_class(name):
    return get_function(name)


def save_model(
    model,
    base_path=None,
    base_name=None,
    evo=None,
    epoch=None,
    iter_=None,
    max_to_keep=None,
):
    name = base_name if base_name is not None else "checkpoint-model"
    name = name + "-evo-{:02d}".format(evo) if evo is not None else name
    name = name + "-epoch-{:04d}".format(epoch) if epoch is not None else name
    name = name + "-iter-{:09d}".format(iter_) if iter_ is not None else name
    name += ".pt"
    path = osp.join(base_path, name)

    torch.save(model.state_dict(), path)

    if max_to_keep is not None:
        base_name = base_name if base_name is not None else "checkpoint-model"
        files = sorted(
            [
                x
                for x in os.listdir(base_path)
                if x.startswith(base_name) and x.endswith(".pt")
            ]
        )

        while len(files) > max_to_keep:
            file_to_be_removed = files[0]
            os.remove(osp.join(base_path, file_to_be_removed))
            del files[0]

    return path


def save_all(
    model,
    optim,
    scheduler=None,
    info_dict=None,
    base_path=None,
    base_name=None,
    evo=None,
    epoch=None,
    iter_=None,
    max_to_keep=None,
):
    name = base_name if base_name is not None else "checkpoint-train"
    name = name + "-evo-{:02d}".format(evo) if evo is not None else name
    name = name + "-epoch-{:04d}".format(epoch) if epoch is not None else name
    name = name + "-iter-{:09d}".format(iter_) if iter_ is not None else name
    name += ".pt"
    path = osp.join(base_path, name)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if info_dict is not None:
        checkpoint.update(info_dict)

    torch.save(checkpoint, path)

    if max_to_keep is not None:
        base_name = base_name if base_name is not None else "checkpoint-train"
        files = sorted(
            [
                x
                for x in os.listdir(base_path)
                if x.startswith(base_name) and x.endswith(".pt")
            ]
        )

        while len(files) > max_to_keep:
            file_to_be_removed = files[0]
            os.remove(osp.join(base_path, file_to_be_removed))
            del files[0]

    return path


def load_model(path, model, strict=True):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=strict)


def load_all(path, model, optim=None, scheduler=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optim is not None:
        optim.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def is_checkpoint(path, base_name=None):
    file = osp.split(path)[1]
    return (
        osp.isfile(path)
        and file.endswith(".pt")
        and (file.startswith(base_name) if base_name is not None else True)
    )


def get_checkpoints(path, base_name=None, include_iter=False):
    if osp.isdir(path):
        checkpoints = [x for x in os.listdir(path)]
        checkpoints = [osp.join(path, checkpoint) for checkpoint in checkpoints]
        checkpoints = [
            checkpoint
            for checkpoint in checkpoints
            if is_checkpoint(checkpoint, base_name)
        ]
    elif osp.isfile(path):
        checkpoints = [path] if is_checkpoint(path) else []
    else:
        checkpoints = []

    if include_iter:
        checkpoints = [
            (iter_from_path(checkpoint), checkpoint) for checkpoint in checkpoints
        ]

    return checkpoints


def iter_from_path(path):
    idx = path.find("-iter-")
    iter_ = int(path[idx + 6 : idx + 6 + 9])
    return iter_


class WeightsOnlySaver:
    def __init__(self, model=None, base_path=None, base_name=None, max_to_keep=None):
        self.model = model
        self.base_path = base_path
        self.base_name = base_name if base_name is not None else "checkpoint-model"
        self.max_to_keep = max_to_keep

    def save(self, evo=None, epoch=None, iter_=None):
        save_path = save_model(
            model=self.model,
            base_path=self.base_path,
            base_name=self.base_name,
            evo=evo,
            epoch=epoch,
            iter_=iter_,
            max_to_keep=self.max_to_keep,
        )

        return save_path

    def get_checkpoints(self, include_iter=False):
        checkpoints = get_checkpoints(
            path=self.base_path, base_name=self.base_name, include_iter=include_iter
        )
        return sorted(checkpoints)

    def get_latest_checkpoint(self, include_iter=False):
        return self.get_checkpoints(include_iter=include_iter)[-1]

    def has_checkpoint(self, path):
        return path in self.get_checkpoints()

    def load(self, full_path=None, strict=True):
        checkpoint = (
            get_checkpoints(path=full_path)[-1]
            if full_path is not None
            else self.get_latest_checkpoint()
        )
        print("Loading checkpoint {} (strict: {}).".format(checkpoint, strict))
        load_model(path=checkpoint, model=self.model, strict=strict)


class TrainStateSaver:
    def __init__(
        self,
        model=None,
        optim=None,
        scheduler=None,
        base_path=None,
        base_name=None,
        max_to_keep=None,
    ):
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.base_path = base_path
        self.base_name = base_name if base_name is not None else "checkpoint-train"
        self.max_to_keep = max_to_keep

    def save(self, info_dict=None, evo=None, epoch=None, iter_=None):
        save_path = save_all(
            model=self.model,
            optim=self.optim,
            scheduler=self.scheduler,
            info_dict=info_dict,
            base_path=self.base_path,
            base_name=self.base_name,
            evo=evo,
            epoch=epoch,
            iter_=iter_,
            max_to_keep=self.max_to_keep,
        )

        return save_path

    def get_checkpoints(self, include_iter=False):
        checkpoints = get_checkpoints(
            path=self.base_path, base_name=self.base_name, include_iter=include_iter
        )
        return sorted(checkpoints)

    def get_latest_checkpoint(self, include_iter=False):
        return self.get_checkpoints(include_iter=include_iter)[-1]

    def has_checkpoint(self, path):
        return path in self.get_checkpoints()

    def load(self, full_path=None):
        checkpoint = (
            get_checkpoints(path=full_path)[-1]
            if full_path is not None
            else self.get_latest_checkpoint()
        )
        print("Loading checkpoint and training state {}).".format(checkpoint))
        out_dict = load_all(
            path=checkpoint,
            model=self.model,
            optim=self.optim,
            scheduler=self.scheduler,
        )
        return out_dict


def warp(
    x, offset=None, grid=None, padding_mode="zeros"
):  # based on PWC-Net Github Repo
    """
    warp an image/tensor according to an offset (optical flow),
    or according to given grid locations.

    Args:
        x: [N, C, H, W] (im2)
        offset: [N, 2, h_out, w_out] offset. h_out/w_out can differ from size H/W of x.
        grid: [N, 2, h_out, w_out] grid of sampling locations.
            h_out/w_out can differ from size H/W of x.
        padding_mode: border or zeros.

    Returns:
        Tuple of:
            [N, C, h_out, w_out] Sampled points from x and
            [N, 1, h_out, w_out] sampling masks.
    """

    N = x.shape[0]
    assert (offset is None and grid is not None) or (
        grid is None and offset is not None
    )

    if offset is not None:
        h_out, w_out = offset.shape[-2:]
        device = x.get_device()

        # create a grid to sample from input image x
        yy, xx = torch.meshgrid(
            torch.arange(h_out), torch.arange(w_out), indexing="ij"
        )  # both (h_out, w_out)
        xx = xx.to(device)
        yy = yy.to(device)
        xx = (xx + 0.5).unsqueeze_(0).unsqueeze_(0)  # 1, 1, h_out, w_out
        yy = (yy + 0.5).unsqueeze_(0).unsqueeze_(0)  # 1, 1, h_out, w_out
        xx = xx.repeat(N, 1, 1, 1)  # N, 1, h_out, w_out
        yy = yy.repeat(N, 1, 1, 1)  # N, 1, h_out, w_out
        grid = torch.cat((xx, yy), 1).float()  # N, 2, h_out, w_out

        # at this point the grid would exactly sample back the image x.
        # now, add the flow as offset to the grid to warp the image
        grid.add_(offset)

    # scale grid to [-1,1]
    h_out, w_out = grid.shape[-2:]
    h_x, w_x = x.shape[-2:]
    grid = grid.permute(0, 2, 3, 1)  # N, h_out, w_out, 2
    xgrid, ygrid = grid.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / w_x - 1
    ygrid = 2 * ygrid / h_x - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)

    # sample from image x given the grid locations to create a new, warped image
    output = nn.functional.grid_sample(
        x, grid, padding_mode=padding_mode, align_corners=False
    )  # N, C, h_out, w_out

    # define how to handle grid selections outside the borders of the image
    if padding_mode == "border":
        # here, selection will be clipped inside the image and all pixels are valid
        mask = torch.ones(
            size=(N, 1, h_out, w_out), device=output.device, requires_grad=False
        )
    else:
        # otherwise all outside pixels are marked as invalid in the mask
        # first, create a mask of ones
        mask = torch.ones(
            size=(N, 1, h_x, w_x), device=output.device, requires_grad=False
        )
        # then, sample from the mask using the grid locations
        # locations outside the image will be 0, inside the image will be 1
        # and inbetween locations will be interpolated
        mask = nn.functional.grid_sample(
            mask, grid, padding_mode="zeros", align_corners=False
        )  # N, 1, h_out, w_out
        # now, only pixels that are sampled completely inside the image should be valid
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

    return output, mask  # N, C, h_out, w_out ; N, 1, h_out, w_out


def shift_multi(x, offsets, padding_mode="zeros"):
    """
    Shifts features x with given integer offsets.

    Args:
        x: Feature map shape (N, C, H, W).
        offsets: Shift offsets shape (S, 2).
        padding_mode: Padding function to use ('zeros' or 'replicate')

    Returns:
        Tuple of:
            Shifted feature maps shape (N, S, C, H, W)
            and masks shape (N, S, H, W).
    """
    offsets_ = offsets.int()
    assert torch.equal(offsets_, offsets)

    dxs, dys = offsets_[:, 0], offsets_[:, 1]
    N, _, H, W = x.shape
    device = x.device
    base_mask = torch.ones(
        size=(N, 1, H, W), dtype=torch.float32, device=device, requires_grad=False
    )

    pad_l, pad_r = max(0, -1 * torch.min(dxs).item()), max(0, torch.max(dxs).item())
    pad_top, pad_bot = max(0, -1 * torch.min(dys).item()), max(0, torch.max(dys).item())
    pad_size = (pad_l, pad_r, pad_top, pad_bot)
    pad_fct = get_pad_fct(padding_mode, pad_size)

    x = pad_fct(x)
    base_mask = pad_fct(base_mask)

    shifteds = []
    masks = []
    for dx, dy in zip(dxs, dys):
        shifted = x[
            :, :, pad_top + dy : pad_top + dy + H, pad_l + dx : pad_l + dx + W
        ]  # N, C, H, W
        mask = base_mask[
            :, :, pad_top + dy : pad_top + dy + H, pad_l + dx : pad_l + dx + W
        ]  # N, 1, H, W
        shifteds.append(shifted)
        masks.append(mask)

    shifteds = torch.stack(shifteds, 1)  # N, S, C, H, W
    masks = torch.cat(masks, 1)  # N, S, H, W

    return shifteds, masks


def get_pad_fct(padding_mode, pad_size):
    if padding_mode == "zeros":
        pad_fct = nn.ConstantPad2d(pad_size, 0)
    elif padding_mode == "repl":
        pad_fct = torch.nn.ReplicationPad2d(pad_size)
    else:
        raise ValueError(f"Padding mode {padding_mode} not supported.")
    return pad_fct
