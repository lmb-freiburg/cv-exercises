import argparse
import os
import os.path as osp
import time
import warnings

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib.augmentation import FlowNetAugmentation
from lib.datasets.flow.flyingchairs import FlyingChairsTrain
from lib.datasets.flow.flyingthings3d import FlyingThings3DTrain
from lib.datasets.flow.sintel import SintelTrain
from lib.flownet import FlowNetC, FlowNetS
from lib.log import Logger
from lib.loss import FlowLoss, PhotometricLoss
from lib.utils import (
    TrainStateSaver,
    WeightsOnlySaver,
    get_checkpoint,
    sample_to_device,
)

PRINT_INTERVAL = 100
LOG_INTERVAL = 5000
LOG_LOSS_INTERVAL = 200
CHECKPOINT_INTERVAL = 100000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", help="Path to folder for output data like logs and model weights."
    )
    parser.add_argument(
        "--iterations", default=600000, type=int, help="Number of training iterations."
    )
    parser.add_argument(
        "--model", default="FlowNetC", help="FlowNetC or FlowNetS model."
    )
    parser.add_argument(
        "--completed_iterations",
        type=int,
        help="Already completed number of iterations.",
    )
    parser.add_argument("--restore", help="Path to a checkpoint to restore from.")
    parser.add_argument(
        "--auto_restore",
        type=str,
        default=None,
        help="Restore checkpoint using path from models.toml file.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading. Set to 0 for debugging.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--dataset",
        default="FlyingThings3D",
        help="Dataset for training. Options: FlyingChairs, FlyingThings3D, Sintel.",
    )
    parser.add_argument(
        "--C",
        default=48,
        type=int,
        help="Base feature dimensionality. Original paper uses C=64. "
        "In the exercise, we use C=48.",
    )
    parser.add_argument(
        "--lr_schedule",
        default="short",
        help="Learning rate schedule (see FlowNet2 paper). "
        "Options: short, long, fine.",
    )
    parser.add_argument(
        "--photometric",
        action="store_true",
        help="Train self-supervised using the photometric loss.",
    )
    parser.add_argument(
        "--smoothness_loss",
        action="store_true",
        help="Use the smoothness loss in addition to the photometric loss.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use: cuda or cpu."
    )
    args = parser.parse_args()

    train(args)


def train(args):
    torch.manual_seed(1)
    np.random.seed(1)

    print_info(args=args)

    out_base = args.output
    checkpoint_dir = osp.join(out_base, "checkpoints")
    log_dir = osp.join(out_base, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    max_iteration = args.iterations
    finished_iterations = (
        0 if args.completed_iterations is None else args.completed_iterations
    )
    epochs = -1

    model = setup_model(args=args)
    dataloader = setup_dataloader(args=args)

    if args.auto_restore is not None:
        args.restore = get_checkpoint(args.auto_restore)

    loss_fct = setup_loss(model=model, args=args)
    optimizer, scheduler = setup_optimization(
        model=model, finished_iterations=finished_iterations, args=args
    )
    saver_all, saver_weights_only = setup_savers(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
    )

    if args.restore is not None:
        restore(
            saver_all=saver_all,
            saver_weights_only=saver_weights_only,
            checkpoint_path=args.restore,
        )

    logger = setup_logger(log_dir=log_dir, model=model, optimizer=optimizer)

    start = time.time()
    print(f"Start training")
    iter_in_epoch = 0
    while finished_iterations < max_iteration:
        epochs += 1
        for iter_in_epoch, sample in enumerate(dataloader):
            sample = sample_to_device(sample, args.device)

            optimizer.zero_grad()
            model_output = run_model(model=model, sample=sample)
            loss, sub_losses, pointwise_losses = loss_fct(
                sample=sample, model_output=model_output
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()

            finished_iterations += 1

            if finished_iterations % PRINT_INTERVAL == 0:
                end = time.time()
                time_per_iteration = (end - start) / PRINT_INTERVAL
                print(
                    "Iteration {}/{} - {:.2f}s per iteration - loss: {:.5f}".format(
                        finished_iterations, max_iteration, time_per_iteration, loss
                    )
                )
                start = time.time()

            if finished_iterations % LOG_INTERVAL == 0:
                logger.log(
                    sample=sample,
                    model_output=model_output,
                    loss=loss,
                    sub_losses=sub_losses,
                    pointwise_losses=pointwise_losses,
                    step=finished_iterations,
                )
            elif finished_iterations % LOG_LOSS_INTERVAL == 0:
                logger.log(
                    sample=sample,
                    model_output=model_output,
                    loss=loss,
                    sub_losses=sub_losses,
                    pointwise_losses=pointwise_losses,
                    step=finished_iterations,
                    loss_only=True,
                )

            if finished_iterations % CHECKPOINT_INTERVAL == 0:
                save_path = saver_all.save(iter_=finished_iterations)
                print("Saving train state to {}.".format(save_path))

            if finished_iterations >= max_iteration:
                save_path = saver_weights_only.save(iter_=finished_iterations)
                print("Saving model weights to {}.".format(save_path))
                break

    epochs += iter_in_epoch / len(dataloader)
    print(
        "Trained for {} iterations (={:.2f} epochs).".format(
            finished_iterations, epochs
        )
    )


def setup_model(args):
    if args.model == "FlowNetC":
        model = FlowNetC(C=args.C)
    elif args.model == "FlowNetS":
        model = FlowNetS(C=args.C)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model = model.to(args.device)
    model.train()

    return model


def setup_dataloader(args):
    if args.dataset == "FlyingThings3D":
        dataset_cls = FlyingThings3DTrain
    elif args.dataset == "FlyingChairs":
        dataset_cls = FlyingChairsTrain
    elif args.dataset == "Sintel":
        dataset_cls = SintelTrain
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    aug_fct = setup_augmentation()
    dataloader = dataset_cls.init_as_loader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        aug_fcts=aug_fct,
    )
    return dataloader


def setup_augmentation():
    aug_fct = FlowNetAugmentation()
    return aug_fct


def setup_loss(model, args):
    if not args.photometric:
        loss_fct = FlowLoss(model)
    else:
        loss_fct = PhotometricLoss(model, use_smoothness_loss=args.smoothness_loss)
    return loss_fct


def setup_optimization(model, finished_iterations, args):
    if args.lr_schedule == "short":
        lr_base = 1e-4
        lr_intervals = [300000, 400000, 500000]
    elif args.lr_schedule == "long":
        lr_base = 1e-4
        lr_intervals = [400000, 600000, 800000, 1000000]
    elif args.lr_schedule == "fine":
        lr_base = 1e-5
        lr_intervals = [1400000, 1500000, 1600000]
    else:
        raise ValueError(f"Unknown lr_schedule: {args.lr_schedule}")

    gamma = 0.5
    params = [p for p in model.parameters() if p.requires_grad]
    print(
        "Total number of parameters to optimize: {}\n".format(
            sum(p.numel() for p in params)
        )
    )
    optimizer = torch.optim.Adam(params, lr=lr_base)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_intervals, gamma=gamma
    )
    with warnings.catch_warnings(record=True):
        # ignore warning "scheduler.step() before optimizer.step()"
        for _ in range(finished_iterations):
            scheduler.step()

    return optimizer, scheduler


def setup_savers(model, optimizer, scheduler, checkpoint_dir):
    saver_all = TrainStateSaver(
        model=model,
        optim=optimizer,
        scheduler=scheduler,
        base_path=checkpoint_dir,
        base_name="checkpoint-train",
        max_to_keep=4,
    )

    saver_weights_only = WeightsOnlySaver(
        model=model, base_path=checkpoint_dir, base_name="checkpoint-model"
    )

    return saver_all, saver_weights_only


def restore(saver_all, saver_weights_only, checkpoint_path):
    if "checkpoint-model" in checkpoint_path:
        saver_weights_only.load(full_path=checkpoint_path)
    else:
        saver_all.load(full_path=checkpoint_path)
    print()


def setup_logger(log_dir, model, optimizer):
    writer = SummaryWriter(log_dir=log_dir, comment="train")
    logger = Logger(writer, model=model, optimizer=optimizer)
    return logger


def run_model(model, sample):
    image_list = sample["images"]
    image_1 = image_list[0]
    image_2 = image_list[1]

    model_output_dict = model(image_1, image_2)

    return model_output_dict


def print_info(args):
    print("Initializing training.")
    print("\tModel: {} (feature dimensionality: {})".format(args.model, args.C))
    print("\tIterations: {}".format(args.iterations))
    print("\tTraining dataset: {}".format(args.dataset))
    print("\tLearning rate schedule: {}".format(args.lr_schedule))
    print("\tSupervised training: {}".format(not args.photometric))

    if args.photometric:
        print("\tUsing smoothness loss: {}".format(args.smoothness_loss))

    if args.restore is not None:
        print("\tRestore from checkpoint: {}".format(args.restore))

    if args.completed_iterations is not None:
        print("\tAlready completed iterations: {}".format(args.completed_iterations))

    print("\tOutput directory: {}".format(args.output))
    print()


if __name__ == "__main__":
    main()
