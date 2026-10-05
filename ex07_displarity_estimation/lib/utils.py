import importlib
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw
import pytoml
from pathlib import Path

from lib.vis import np3d
from scipy.spatial.transform import Rotation


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


def get_function(
    name,
):  # from https://github.com/aschampion/diluvian/blob/master/diluvian/util.py
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
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=strict
    )


def load_all(path, model, optim=None, scheduler=None):
    checkpoint = torch.load(path)
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
        print("Loading checkpoint {}).".format(checkpoint))
        out_dict = load_all(
            path=checkpoint,
            model=self.model,
            optim=self.optim,
            scheduler=self.scheduler,
        )
        return out_dict


def warp(
    x, offset=None, grid=None, padding_mode="border"
):  # based on PWC-Net Github Repo
    """Samples from a tensor according to an offset (e.g. optical flow), or according to given grid locations.

    Input can either be an offset or a grid of absolute sampling locations.

    Args:
        x: Input tensor with shape (N,C,H,W).
        offset: Offset with shape (N,2,h_out,w_out) where h_out can w_out can differ from size H and W of x.
        grid: Grid of absolute sampling locations with shape (N,2,h_out,w_out) where h_out can w_out can
         differ from size H and W of x.
        padding_mode: Padding for sampling out of the image border. Can be 'border' or 'zeros'.
    Returns:
        Tuple (output, mask) containing
            output: Sampled points from x with shape (N,C,h_out,w_out).
            mask: Sampling mask with shape (N,1,h_out,w_out).
    """

    N = x.shape[0]

    assert (offset is None and grid is not None) or (
        grid is None and offset is not None
    )

    if offset is not None:
        h_out, w_out = offset.shape[-2:]
        device = x.get_device()

        yy, xx = torch.meshgrid(
            torch.arange(h_out), torch.arange(w_out), indexing="ij"
        )  # both (h_out, w_out)
        xx = xx.to(device)
        yy = yy.to(device)
        xx = (xx + 0.5).unsqueeze_(0).unsqueeze_(0)  # 1, 1, h_out, w_out
        yy = (yy + 0.5).unsqueeze_(0).unsqueeze_(0)  # 1, 1, h_out, w_out
        xx = xx.repeat(N, 1, 1, 1)  # TODO: maybe this can be removed?
        yy = yy.repeat(N, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()  # N, 2, h_out, w_out

        grid.add_(offset)

    # scale grid to [-1,1]
    h_out, w_out = grid.shape[-2:]
    h_x, w_x = x.shape[-2:]

    grid = grid.permute(0, 2, 3, 1)  # N, h_out, w_out, 2
    xgrid, ygrid = grid.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / w_x - 1
    ygrid = 2 * ygrid / h_x - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)

    output = nn.functional.grid_sample(
        x, grid, padding_mode=padding_mode, align_corners=False
    )  # N, C, h_out, w_out

    if padding_mode == "border":
        mask = torch.ones(
            size=(N, 1, h_out, w_out), device=output.device, requires_grad=False
        )

    else:
        mask = torch.ones(
            size=(N, 1, h_x, w_x), device=output.device, requires_grad=False
        )
        mask = nn.functional.grid_sample(
            mask, grid, padding_mode="zeros", align_corners=False
        )  # N, 1, h_out, w_out
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

    return output, mask  # N, C, h_out, w_out ; N, 1, h_out, w_out


def shift_multi(x, offsets, padding_mode="zeros"):
    """Shifts a tensor x with given integer offsets.

    Args:
        x: Input tensor with shape (N,C,H,W).
        offsets: Shift offsets with shape (S,2).
        padding_mode: Padding for shifting out of the image border. Can be 'border' or 'zeros'.

    Returns:
        Tuple (shifteds, masks) containing
            shifteds: Shifted tensor x with shape (N,S,C,H,W).
            masks: Masks with shape (N,S,H,W). Masks are 0 for shifts out of the image border when
              padding_mode is 'zeros', otherwise 1.
    """

    offsets_ = offsets.int()
    assert torch.all(offsets_ == offsets)

    dxs, dys = offsets_[:, 0], offsets_[:, 1]
    N, _, H, W = x.shape
    device = x.device
    base_mask = torch.ones(
        size=(N, 1, H, W), dtype=torch.float32, device=device, requires_grad=False
    )

    pad_l, pad_r = max(0, -1 * torch.min(dxs).item()), max(0, torch.max(dxs).item())
    pad_top, pad_bot = max(0, -1 * torch.min(dys).item()), max(0, torch.max(dys).item())
    pad_size = (pad_l, pad_r, pad_top, pad_bot)
    pad_fct = (
        nn.ConstantPad2d(pad_size, 0)
        if padding_mode == "zeros"
        else torch.nn.ReplicationPad2d(pad_size)
    )
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


def transform_from_rot_trans(R, t):
    """Computes a transformation matrix from a rotation matrix and a translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def trans_from_transform(T):
    """Computes a translation vector from a transformation matrix."""
    return T[0:3, 3]


def rot_from_transform(T):
    """Computes a rotation matrix from a transformation matrix."""
    return T[0:3, 0:3]


def invert_transform(T):
    """Inverts a transformation matrix."""
    R = rot_from_transform(T)
    t = trans_from_transform(T)
    R_inv = R.T
    t_inv = np.dot(-R.T, t)
    return transform_from_rot_trans(R_inv, t_inv)


def identity_transform():
    """Initializes an identity transformation matrix."""
    R = np.eye(3)
    t = np.zeros(3)
    return transform_from_rot_trans(R, t)


def angleaxis_from_rot(R):
    """Computes an angleaxis rotation representation from a rotation matrix.

    An angleaxis representation is a vector where the vector direction represents the rotation axis and the vector
    magnitude represents the rotation angle (in rad).
    """
    return Rotation.from_matrix(R).as_rotvec()


def angle_axis_from_angleaxis(angleaxis, eps=1e-6):
    """Computes an angle, axis rotation representation from a rotation matrix.

    An angle, axis representation is an axis vector with magnitude 1 whose direction represents the rotation axis and
    an angle scalar that represents the rotation angle (in rad).
    """
    angle = np.linalg.norm(angleaxis)

    if angle < eps:
        angle = 0
        axis = np.zeros(3)
    else:
        axis = angleaxis / angle

    return angle, axis


def angle_axis_from_rot(R, eps=1e-6):
    """Computes an angle, axis rotation representation from a rotation matrix."""
    return angle_axis_from_angleaxis(angleaxis_from_rot(R), eps=eps)


def rot_from_angleaxis(angleaxis):
    """Computes a rotation matrix from an angleaxis representation."""
    return Rotation.from_rotvec(angleaxis).as_matrix()


def rot_from_angle_axis(angle, axis):
    """Computes a rotation matrix from an angle, axis representation."""
    angleaxis = angle * axis
    return rot_from_angleaxis(angleaxis)


def rot_x(t):
    """Computes a rotation matrix for a rotation around the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(t):
    """Computes a rotation matrix for a rotation around the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(t):
    """Computes a rotation matrix for a rotation around the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def transform_from_rot_trans_2d(R, t):
    """Computes a 2d transformation matrix from a rotation matrix and a translation vector."""
    R = R.reshape(2, 2)
    t = t.reshape(2, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 1]))


def cross_mat_from_vec(vec):
    """Computes a matrix from a vector to express a cross product with that vector by a dot product with the matrix."""
    vec_x = np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
    return vec_x


def project_to_image(X_ref, K, to_ref_transform, return_hom=False):
    """Projects a 3d point given in a reference coordinate system to
    image coordinates of the current image.

    Args:
        X_ref: (X,Y,Z) coordinates of the 3d point relative to a reference coordinate system.
        K: Camera intrinsics of shape (3, 3).
        to_ref_transform: Transformation matrix from the current to the reference camera,
            shape (4, 4). Note: this is the transform that maps from the current to the reference
            coordinate system and that maps a point given in coordinates of the reference camera
            to coordinates of the current camera, e.g. p_in_cur = to_ref_transform.dot(p_in_ref).
        return_hom: Boolean flag that indicates whether the image coordinates
            should be returned in homogeneous form.

    Returns:
        Image coordinates of shape (2) if hom=False and shape (3) if hom=True.
    """
    # START TODO #################
    # Outline:
    # 1. Compute the camera matrix P from K and to_ref_transform. See class 10, camera calibration.
    # Hint: Discard the last row of to_ref_transform to get the external parameters M=(R|t).
    # 2. Transform X_ref to homogeneous coordinates X_ref_hom by appending 1 using np.append.
    # 3. Compute the homogeneous image coordinates x_hom by taking the dot product between P and X_ref_hom.
    # 4. Compute x_hom by dividing x_hom[0] / x_hom[2] and x_hom[1] / x_hom[2].
    # 5. Return x if return_hom is False, otherwise return x_hom.
    raise NotImplementedError
    # END TODO ###################


def get_epipole(K, to_ref_transform, return_hom=False):
    """Gets epipole coordinates from reference camera in current image.

    The epipole is computed by projecting the center of the reference camera which is (0, 0, 0)
    to the current image.

    Args:
        K: Camera intrinsics of shape (3, 3).
        to_ref_transform: Transformation matrix from the current to the reference camera,
            shape (4, 4). Note: this is the transform that maps from the current to the reference
            coordinate system and that maps a point given in coordinates of the reference camera
            to coordinates of the current camera, e.g. p_in_cur = to_ref_transform.dot(p_in_ref).
        return_hom: Boolean flag that indicates whether the image coordinates
            should be returned in homogeneous form.


    Returns:
        Epipole of shape (2) if hom=False and shape (3) if hom=True.
    """
    # START TODO #################
    # Outline: construct a point (0, 0, 0) and use the function project_to_image to project it
    # to the current image (pass the hom argument to return_hom of project_to_image).
    raise NotImplementedError
    # END TODO ###################


def compute_essential_matrix(to_ref_transform):
    """Computes the essential matrix from a reference camera to current camera.

    The essential matrix is computed via E_ref_to_cur = tx * R.
    t is the translation from the current to the reference camera.
    t_x is the skew-symmetric representation of t that can be used for a cross product with t.
    R is the rotation from the current to the reference camera.

    Args:
        to_ref_transform: Transformation matrix from the current to the reference camera,
            shape (4, 4). Note: this is the transform that maps from the current to the reference
            coordinate system and that maps a point given in coordinates of the reference camera
            to coordinates of the current camera, e.g. p_in_cur = to_ref_transform.dot(p_in_ref).

    Returns:
        Essential matrix from reference camera to current camera.
    """
    # START TODO #################
    # Outline:
    # 1. Compute t, R, tx using the functions trans_from_transform, rot_from_transform,
    # cross_mat_from_vec from above.
    # 2. Compute E_ref_to_cur as dot product from tx and R.
    raise NotImplementedError
    # END TODO ###################


def compute_fundamental_matrix(K, to_ref_transform):
    """Computes the fundamental matrix from a reference camera to current camera.

    The fundamental matrix is computed via F_ref_to_cur = e_cur_x * P_cur * P_ref_inv.
    e_cur is the epipole of the reference camera in the current image.
    e_cur_x is the skew-symmetric representation of
    e_cur that can be used for a cross product with e_cur.

    Args:
        K: Camera intrinsics of shape (3, 3).
        to_ref_transform: Transformation matrix from the current to the reference camera,
            shape (4, 4). Note: this is the transform that maps from the current to the reference
            coordinate system and that maps a point given in coordinates of the reference camera
            to coordinates of the current camera, e.g. p_in_cur = to_ref_transform.dot(p_in_ref).

    Returns:
        Fundamental matrix from reference camera to current camera.
    """
    # START TODO #################
    # Outline:
    # 1. Compute epi_from_ref_in_cur_x using the functions get_epipole and cross_mat_from_vec.
    # 2. Compute P_cur from K and to_ref_transform.
    # 3. Compute P_ref from K and an identity transform.
    # 4. Invert P_ref using np.linalg.pinv
    # 5. Compute F as dot product from epi_from_ref_in_cur_x, P_cur and P_ref_inv.
    raise NotImplementedError
    # END TODO ###################


def plot_epipolar_line(image_cur, F_ref_to_cur, x_ref, line_color=(255, 0, 0)):
    image_cur = np3d(image_cur, image_range_text_off=True)

    x_ref = np.append(np.array(x_ref), 1)
    l_cur = np.dot(F_ref_to_cur, x_ref)
    x_0 = -1
    y_0 = (-l_cur[2] - (x_0 * l_cur[0])) / l_cur[1]
    x_1 = image_cur.width + 1
    y_1 = (-l_cur[2] - (x_1 * l_cur[0])) / l_cur[1]

    draw = ImageDraw.Draw(image_cur)
    draw.line((x_0, y_0) + (x_1, y_1), fill=line_color, width=4)
    return np.array(image_cur)


def rectify_images(image_l, image_r, K, r_to_l_transform):
    """Rectifies two images with a known camera calibration using bouquet's algorithm.

    This algorithm is also applied in the OpenCV function cvStereoRectify().
    A description of the algorithm can be found in the book
    Learning OpenCV by Gary Bradski and Adrian Kaehler, O'Reilly, page 433
    ( https://www.bogotobogo.com/cplusplus/files/OReilly%20Learning%20OpenCV.pdf ).
    Here, we use a similar notation as in the book.

    Args:
        image_l: Left image of shape (3, H, W).
        image_r: Right image of shape (3, H, W).
        K: Camera intrinsics of shape (3, 3).
        r_to_l_transform: Transformation matrix shape (4, 4) from the right to the left camera.
            Note: this is the transform that maps from the right to the left coordinate system
            and that maps a point given in coordinates of the left camera to coordinates
            of the right camera, e.g. p_in_r = r_to_l_transform.dot(p_in_l).

    Returns:
        Tuple (image_l_rect, image_r_rect, H_l, H_r, rrect_to_lrect_transform) containing
            image_l_rect: Rectified left image.
            image_r_rect: Rectified right image.
            H_l: Homography that was applied to the left image for rectification.
            H_r: Homography that was applied to the right image for rectification.
            rrect_to_lrect_transform: Transformation matrix from the right to the left
                rectified camera.
    """
    # 1. Rotate right camera and left camera in 3d such that they have the same direction:
    # To do so, transfer the known rotation between the cameras
    # to an angle, axis representation
    # (use functions rot_from_transform, angle_axis_from_rot, etc. from above).
    # Then create a rotation matrix that rotates both cameras around the axis for +/- angle/2.
    # START TODO ###################
    raise NotImplementedError
    # END TODO ###################

    # 2. Rotate cameras (in 3d) such that epipole goes to infinity:
    # To do so, construct a rotation matrix R_rect such that the baseline (=translation vector)
    # between the cameras is transformed to a vector of the form (translation_norm, 0, 0).
    # The construction of R_rect is described in the book mentioned above.
    # START TODO ###################
    raise NotImplementedError
    # END TODO ###################
    R_rect = np.stack([e1, e2, e3])

    # 3. Combine the rotations r_r, r_l, R_rect to get rotation matrices
    # R_l and R_r for both cameras:
    # R_l = R_rect (matmul) r_l
    # R_r = R_rect (matmul) r_r
    # START TODO ###################
    raise NotImplementedError
    # END TODO ###################

    # A rotation R in 3d leads to a homography H = K*R*K^-1 which can be
    # applied to the image to warp it to the rotated viewing direction.
    # 4. Compute the homographies H_l and H_r for the left and right images:
    # You can use np.linalg.inv() to invert K.
    # START TODO ###################
    raise NotImplementedError
    # END TODO ###################

    # 5. Compute the transformation rrect_to_lrect_transform between the rectified cameras:
    # R_l is actually a rotation lrect_to_l_rotation and R_r is rrect_to_r_rotation.
    # Hence it is:
    # rrect_to_lrect_transform = rrect_to_r_transform * r_to_l_transform * lrect_to_l_transform^-1
    lrect_to_l_transform = transform_from_rot_trans(R_l, np.zeros(3))
    rrect_to_r_transform = transform_from_rot_trans(R_r, np.zeros(3))
    rrect_to_lrect_transform = rrect_to_r_transform.dot(r_to_l_transform).dot(
        invert_transform(lrect_to_l_transform)
    )
    # Alternatively, we could exploit that we know that
    # a) there is no rotation between the rectified cameras
    # b) there is only a x-translation between the rectified cameras
    # Hence:
    # rrect_to_lrect_transform = transform_from_rot_trans(
    #     np.eye(3), np.array([-np.linalg.norm(T), 0, 0]))

    # 6. Rectify the left and right images by warping with the homographies:
    import cv2

    image_l_rect = cv2.warpPerspective(
        image_l.transpose([1, 2, 0]), H_l, image_l.shape[-2:][::-1]
    )
    image_r_rect = cv2.warpPerspective(
        image_r.transpose([1, 2, 0]), H_r, image_r.shape[-2:][::-1]
    )
    image_l_rect = image_l_rect.transpose([2, 0, 1])
    image_r_rect = image_r_rect.transpose([2, 0, 1])

    return image_l_rect, image_r_rect, H_l, H_r, rrect_to_lrect_transform


def rectify_images_with_opencv(image_l, image_r, K, r_to_l_transform):
    # this function can be ignored for the exercise
    import cv2

    distCoeff = np.zeros(4, dtype=np.float64)
    R = rot_from_transform(r_to_l_transform).astype(np.float64)
    t = trans_from_transform(r_to_l_transform).astype(np.float64)

    image_size = (image_l.shape[-1], image_l.shape[-2])

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=K,
        distCoeffs1=distCoeff,
        cameraMatrix2=K,
        distCoeffs2=distCoeff,
        imageSize=image_size,
        R=R,
        T=t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=1,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        cameraMatrix=K,
        distCoeffs=distCoeff,
        R=R1,
        newCameraMatrix=P1,
        size=image_size,
        m1type=cv2.CV_32FC1,
    )

    map2x, map2y = cv2.initUndistortRectifyMap(
        cameraMatrix=K,
        distCoeffs=distCoeff,
        R=R2,
        newCameraMatrix=P2,
        size=image_size,
        m1type=cv2.CV_32FC1,
    )

    image_l_rect = cv2.remap(
        image_l.transpose([1, 2, 0]), map1x, map1y, cv2.INTER_LINEAR
    )
    image_r_rect = cv2.remap(
        image_r.transpose([1, 2, 0]), map2x, map2y, cv2.INTER_LINEAR
    )

    image_l_rect = image_l_rect.transpose([2, 0, 1])
    image_r_rect = image_r_rect.transpose([2, 0, 1])

    return image_l_rect, image_r_rect
