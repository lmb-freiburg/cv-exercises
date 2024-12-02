import argparse
import os
import os.path as osp
import time

import numpy as np
import pandas
import torch
from torch.utils.tensorboard import SummaryWriter

from lib.augmentation import FlowNetAugmentation
from lib.datasets.disp.flyingthings3d import FlyingThings3DTest
from lib.dispnet import DispNetC, DispNetS
from lib.flownet import FlowNetC, FlowNetS
from lib.log import Logger
from lib.metrics import compute_disp_metrics
from lib.utils import load_model, load_all, get_checkpoint, sample_to_device

PRINT_INTERVAL = 100
LOG_INTERVAL = 100


def main():
    args = setup_args()
    torch.manual_seed(1)
    np.random.seed(1)
    evaluate(args)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Path to folder for evaluation outputs.")
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
        "--dataset",
        default="FlyingThings3D",
        help="Dataset for evaluation. Options: FlyingThings3D.",
    )
    parser.add_argument(
        "--C",
        default=48,
        type=int,
        help="Base feature dimensionality. Original paper uses C=64. "
        "In the exercise we use C=48.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use: cuda or cpu."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading. Set to 0 for debugging.",
    )
    args = parser.parse_args()
    return args


def evaluate(args):
    if args.auto_restore is not None:
        args.restore = get_checkpoint(args.auto_restore)
    print_info(args=args)

    out_base = args.output
    eval_base = osp.join(out_base, "eval")
    eval_dir = osp.join(eval_base, args.dataset)
    log_dir = osp.join(eval_dir, "logs")
    os.makedirs(eval_base, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = setup_model(args=args)
    dataloader = setup_dataloader(args=args)

    if args.restore is not None:
        restore(model=model, checkpoint_path=args.restore)

    logger = setup_logger(log_dir=log_dir, model=model)

    metrics = pandas.DataFrame()
    metrics.index.name = "sample"
    metrics.columns.name = "metric"

    start = time.time()
    for sample_idx, sample in enumerate(dataloader):
        sample = sample_to_device(sample, args.device)

        model_output = run_model(model=model, sample=sample)

        metric_values, qualitatives = compute_disp_metrics(
            sample=sample, model_output=model_output
        )
        log_metrics(metrics=metrics, values=metric_values, sample_idx=sample_idx)

        if (sample_idx > 0) and (sample_idx % PRINT_INTERVAL == 0):
            end = time.time()
            time_per_iteration = (end - start) / PRINT_INTERVAL
            print(
                "Sample {}/{} - {:.2f}s per sample - metrics:".format(
                    sample_idx, len(dataloader), time_per_iteration
                )
            )
            print("\t" + metrics.loc[sample_idx].to_string().replace("\n", "\n\t"))
            start = time.time()

        if sample_idx % LOG_INTERVAL == 0:
            logger.log_eval(
                sample=sample,
                model_output=model_output,
                metrics=metric_values,
                qualitatives=qualitatives,
                sample_idx=sample_idx,
            )
        del metric_values, qualitatives, sample, model_output

    results = metrics.mean()
    metrics.to_csv(osp.join(eval_dir, "metrics.csv"))
    results.to_csv(osp.join(eval_dir, "results.csv"))
    print()
    print(f"Saved results to {eval_dir}")
    print(results.to_string())

    return metrics, results


def setup_model(args):
    if args.model == "FlowNetC":
        model = FlowNetC(C=args.C)
    elif args.model == "FlowNetS":
        model = FlowNetS(C=args.C)
    elif args.model == "DispNetC":
        model = DispNetC(C=args.C)
    elif args.model == "DispNetS":
        model = DispNetS(C=args.C)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model = model.to(args.device)
    model.eval()

    return model


def setup_dataloader(args):
    global LOG_INTERVAL

    if args.dataset == "FlyingThings3D":
        dataset_cls = FlyingThings3DTest
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    preprocess_fct = setup_preprocessing()
    dataloader = dataset_cls.init_as_loader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        aug_fcts=preprocess_fct,
    )
    return dataloader


def setup_preprocessing():
    preprocess_fct = FlowNetAugmentation()
    preprocess_fct.out_size = (
        None  # upscale to next height and width that are divisible by 64
    )
    preprocess_fct.spatial_aug_prob = 0.0  # only resize, no cropping
    preprocess_fct.color_aug_prob = 0.0  # no color augmentation
    preprocess_fct.augment_image_only = True  # do not resize the ground truth
    return preprocess_fct


def restore(model, checkpoint_path):
    print("Restoring model weights from {}".format(checkpoint_path))
    if "checkpoint-train" in checkpoint_path:
        load_all(path=checkpoint_path, model=model)
    else:
        load_model(path=checkpoint_path, model=model, strict=True)
    print()


def setup_logger(log_dir, model):
    writer = SummaryWriter(log_dir=log_dir, comment="test")
    logger = Logger(writer, model=model)
    return logger


def run_model(model, sample):
    image_list = sample["images"]
    image_left = image_list[0]
    image_right = image_list[1]

    model_output_dict = model(image_left, image_right)

    return model_output_dict


def log_metrics(metrics, values, sample_idx):
    for k, v in values.items():
        metrics.loc[sample_idx, k] = v


def print_info(args):
    print("Initializing evaluation.")
    print("\tModel: {} (feature dimensionality: {})".format(args.model, args.C))
    print("\tEvaluation dataset: {}".format(args.dataset))

    if args.restore is not None:
        print("\tRestore from checkpoint: {}".format(args.restore))
    else:
        print(
            "\tWarning: No checkpoint given. Evaluation is done with an untrained model!"
        )

    print("\tOutput directory: {}".format(args.output))
    print()


if __name__ == "__main__":
    main()
