import argparse
import copy
import random
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights

CORRUPTIONS = [
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of data loader threads per dataloader. "
             "0 will use the main thread and is good for debugging",
    )
    parser.add_argument(
        "--num_bn_updates", type=int, default=10, help="number of batch norm updates"
    )
    parser.add_argument("--corruption", type=str, default="glass_blur")
    parser.add_argument("--severity", type=int, default=1)
    parser.add_argument("--apply_bn", action="store_true")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="run a full validation on all corruptions and all severities",
    )
    parser.add_argument(
        "--clean_eval",
        action="store_true",
        help="validate the model on the clean uncorrutped dataset",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/project/cv-ws2425/lmb/data/data",
        help="where to load the data from",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # seed everything
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Seed set: {seed}")

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running training on: {device}")

    # use pre trained resnet18
    model: models.resnet.ResNet = models.resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    )
    print(model)
    print(type(model))

    # send model to device
    model = model.to(device)

    # define transformation
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Evaluation on the uncorrupted clean dataset
    if args.clean_eval:
        print("clean validation")
        val_dataset = datasets.ImageFolder(
            f"{args.data_dir}/imagenet200", transform=val_transform
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        loss, acc1 = validate(
            model, val_loader, device, args.apply_bn, args.num_bn_updates
        )
        print(f"Validation complete. Loss {loss:.6f} accuracy {acc1:.2%}")

    elif args.evaluate:
        # validate all corruptions

        corruption_accs = validate_all_c(
            args.data_dir,
            model,
            val_transform,
            device,
            args.batch_size,
            args.apply_bn,
            args.num_bn_updates,
        )
        pprint(corruption_accs)

    else:
        # validate on corruption given by user or default
        validate_c(
            args.data_dir,
            model,
            val_transform,
            device,
            args.batch_size,
            args.apply_bn,
            args.num_bn_updates,
            args.corruption,
            args.severity,
        )


def validate(model, val_loader, device, apply_bn, num_bn_updates):
    # Validate the model on a given dataset

    if apply_bn:
        model = update_bn_params(model, val_loader, device, num_bn_updates)

    model.eval()
    with torch.no_grad():
        total_loss, total_acc, num_batches = 0.0, 0.0, 0
        for batch_idx, (data, label) in enumerate(val_loader):
            num_batches += 1
            # move data to our device
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(output, label)
            pred = torch.argmax(output, dim=1)
            correct = torch.sum(pred == label)
            acc = correct / len(data)
            total_loss += loss.item()
            total_acc += acc.item()

            # log the loss
            if batch_idx % 100 == 0:
                print(f"Validation step {batch_idx}/{len(val_loader)}")
    # normalize accuracy and loss over the number of batches
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    # print(f"Validation complete. Loss {avg_loss:.6f} accuracy {avg_acc:.2%}")

    return avg_loss, avg_acc


def validate_c(
    data_dir,
    model,
    val_transform,
    device,
    batch_size,
    apply_bn,
    num_bn_updates,
    corruption,
    severity,
):
    # validate the model on a given corrupted dataset

    print(f"{corruption} severity {severity}")
    valdir = Path(data_dir) / "dataset_folder" / corruption / str(severity)
    val_dataset = datasets.ImageFolder(valdir.as_posix(), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    loss, acc1 = validate(model, val_loader, device, apply_bn, num_bn_updates)
    print(f"Validation complete. Loss {loss:.6f} accuracy {acc1:.2%}")

    return


def validate_all_c(
    data_dir, model, val_transform, device, batch_size, apply_bn, num_bn_updates
):
    # validate the model on the full corrupted dataset
    corruption_accs = {}
    for c in CORRUPTIONS:
        print(c)
        for s in range(1, 4):
            valdir = Path(data_dir) / "dataset_folder" / c / str(s)
            val_dataset = datasets.ImageFolder(
                valdir.as_posix(), transform=val_transform
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
            loss, acc1 = validate(model, val_loader, device, apply_bn, num_bn_updates)
            if c in corruption_accs:
                corruption_accs[c].append(acc1 * 100)
            else:
                corruption_accs[c] = [acc1 * 100]
            print(
                f"Validation of {c} severity {s} complete. Loss {loss:.6f} accuracy {acc1:.2%}"
            )

    return corruption_accs


def update_bn_params(model, val_loader, device, num_bn_updates):
    # START TODO #################
    # 1. access the dataset as val_loader.dataset and create a new dataloader, with shuffling
    # 2. create a copy of the model
    # 3. set the entire model to evaluation mode, except set the batchnorm modules to train mode
    # 4. run forward passes to update batchnorm statistics

    val_loader = torch.utils.data.DataLoader(
        val_loader.dataset,
        batch_size=val_loader.batch_size,
        shuffle=True,
        num_workers=val_loader.num_workers,
    )

    def use_test_statistics(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.train()

    model = copy.deepcopy(model)
    model.eval()
    model.apply(use_test_statistics)
    print("Updating BN params (num updates:{})".format(num_bn_updates))
    with torch.no_grad():
        for i, (images, label) in enumerate(val_loader):
            if i >= num_bn_updates:
                break
            images = images.to(device, non_blocking=True)
            _output = model(images)
    print("Done.")

    # END TODO ##################
    return model


if __name__ == "__main__":
    main()
