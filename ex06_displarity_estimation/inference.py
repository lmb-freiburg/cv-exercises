import argparse
import math
import os.path as osp

import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling

from eval import setup_model
from lib.utils import load_model, load_all, rectify_images, get_checkpoint
from lib.vis import np2d


def main():
    args = setup_args()
    torch.manual_seed(1)
    np.random.seed(1)
    inference(args)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        help="Path to folder with input images. "
        "Output will be written to the same folder.",
    )
    parser.add_argument(
        "--model",
        default="DispNetC",
        help="DispNetC, DispNetC, FlowNetC or FlowNetS model.",
    )
    parser.add_argument("--restore", help="Path to a checkpoint to restore from.")
    parser.add_argument(
        "--auto_restore",
        type=str,
        default=None,
        help="Restore checkpoint using path from models.toml file.",
    )
    parser.add_argument(
        "--C",
        default=48,
        type=int,
        help="Base feature dimensionality. Original paper uses C=64. "
        "In the exercise we use C=48.",
    )
    parser.add_argument(
        "--rectify",
        action="store_true",
        help="Rectify the input images. Requires the relative pose and intrinsics.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use: cuda or cpu."
    )
    args = parser.parse_args()
    return args


def inference(args):
    if args.auto_restore is not None:
        args.restore = get_checkpoint(args.auto_restore)
    print_info(args=args)

    model = setup_model(args=args)
    sample = load_data(path=args.path, rectify=args.rectify)

    if args.restore is not None:
        restore(model=model, checkpoint_path=args.restore)

    model_output = run_model(model=model, sample=sample)
    write_pred(sample=sample, model_output=model_output, path=args.path)

    print("Done. Output written to {}.".format(osp.join(args.path)))
    return sample, model_output


def preprocess_image(image):
    w_orig, h_orig = image.size
    h_in = int(math.ceil(h_orig / 64.0) * 64.0)
    w_in = int(math.ceil(w_orig / 64.0) * 64.0)

    image = image.resize((w_in, h_in), Resampling.BILINEAR)
    image = np.array(image)
    image = ((image / 255.0) - 0.4).astype(np.float32)
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, 0)  # 1, 3, H, W

    return image


def load_data(path, rectify=False):
    path_left = osp.join(path, "left.png")
    path_right = osp.join(path, "right.png")

    image_left = Image.open(path_left)
    image_right = Image.open(path_right)
    w_orig, h_orig = image_left.size

    sample = {}

    K = (
        np.load(osp.join(path, "K.npy"))
        if osp.isfile(osp.join(path, "K.npy"))
        else None
    )
    right_to_left_transform = (
        np.load(osp.join(path, "right_to_left_transform.npy"))
        if osp.isfile(osp.join(path, "right_to_left_transform.npy"))
        else None
    )

    if rectify:
        image_left = np.array(Image.open(path_left)).transpose([2, 0, 1])  # 3, H, W
        image_right = np.array(Image.open(path_right)).transpose([2, 0, 1])  # 3, H, W

        image_left, image_right, H_l, H_r, right_to_left_transform = rectify_images(
            image_left, image_right, K, right_to_left_transform
        )

        image_left = Image.fromarray(image_left.transpose([1, 2, 0]))
        image_right = Image.fromarray(image_right.transpose([1, 2, 0]))
        sample["H_l"] = H_l
        sample["H_r"] = H_r

    image_left = preprocess_image(image_left)
    image_right = preprocess_image(image_right)

    sample["images"] = [image_left, image_right]
    sample["w_orig"] = w_orig
    sample["h_orig"] = h_orig
    if right_to_left_transform is not None:
        sample["right_to_left_transform"] = right_to_left_transform
    if K is not None:
        sample["K"] = K
    return sample


def restore(model, checkpoint_path):
    print("Restoring model weights from {}".format(checkpoint_path))
    if "checkpoint-train" in checkpoint_path:
        load_all(path=checkpoint_path, model=model)
    else:
        load_model(path=checkpoint_path, model=model, strict=True)
    print()


def data_to_torch(data, device=None):
    if isinstance(data, dict):
        return {key: data_to_torch(data[key], device) for key in data.keys()}
    elif isinstance(data, list):
        return [data_to_torch(val, device) for val in data]
    elif isinstance(data, np.ndarray):
        return data_to_torch(torch.from_numpy(data).float())
    elif isinstance(data, torch.Tensor):
        return data.cuda(device=device)
    else:
        return data


def data_to_numpy(data):
    if isinstance(data, dict):
        return {key: data_to_numpy(data[key]) for key in data.keys()}
    elif isinstance(data, list):
        return [data_to_numpy(val) for val in data]
    elif isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()
    else:
        return data


def run_model(model, sample):
    sample = data_to_torch(sample)
    image_list = sample["images"]
    image_left = image_list[0]
    image_right = image_list[1]

    model_output_dict = model(image_left, image_right)
    model_output_dict = data_to_numpy(model_output_dict)

    return model_output_dict


def write_pred(sample, model_output, path):
    w_orig = sample["w_orig"]
    h_orig = sample["h_orig"]

    pred_disp = (
        model_output["pred_disp"]
        if "pred_disp" in model_output
        else model_output["pred_flow"][:, 0:1, :, :]
    )
    pred_disp = np.array(
        Image.fromarray(pred_disp.squeeze()).resize(
            (w_orig, h_orig), Resampling.NEAREST
        )
    ).astype(np.float32)

    if "right_to_left_transform" in sample and "K" in sample:
        right_to_left_transform = sample["right_to_left_transform"]
        K = sample["K"]
        baseline = right_to_left_transform[0, 3]
        f = K[0, 0]
        pred_depth = np.clip(f * baseline / (pred_disp - 1e-2), 0, 100)
    else:
        pred_depth = None

    if "H_l" in sample:
        import cv2

        H_l = sample["H_l"]
        pred_disp = cv2.warpPerspective(
            pred_disp,
            np.linalg.inv(H_l),
            pred_disp.shape[::-1],
            flags=cv2.INTER_NEAREST,
        )
        if pred_depth is not None:
            pred_depth = cv2.warpPerspective(
                pred_depth,
                np.linalg.inv(H_l),
                pred_depth.shape[::-1],
                flags=cv2.INTER_NEAREST,
            )

    np.save(osp.join(path, "pred_disp.npy"), pred_disp)
    np2d(pred_disp).save(osp.join(path, "pred_disp.png"))
    if pred_depth is not None:
        np.save(osp.join(path, "pred_depth.npy"), pred_depth)
        np2d(pred_depth).save(osp.join(path, "pred_depth.png"))


def print_info(args):
    print("Initializing inference.")
    print("\tModel: {} (feature dimensionality: {})".format(args.model, args.C))
    print("\tData path: {}".format(args.path))

    if args.restore is not None:
        print("\tRestore from checkpoint: {}".format(args.restore))
    else:
        print(
            "\tWarning: No checkpoint given. Inference is done with an untrained model!"
        )

    print()


if __name__ == "__main__":
    main()
