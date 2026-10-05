"""Simple example script."""

import argparse
import os

import torch
import torch as th
import torchvision
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from lib.augmentations import horizontal_flip, random_resize_crop
from lib.cifar_dataset import get_dataloaders
from lib.lenet_model import LeNet


def get_transforms(args):
    if args.transforms == 'basic':
        train_transforms = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif args.transforms == 'own':
        # START TODO #################
        # use torchvision.transforms.Compose to compose our custom augmentations
        # horizontal_flip, random_resize_crop, ToTensor, Normalize
        # you can play around with the parameters
        # train_transforms=
        raise NotImplementedError
        # END TODO #################
    elif args.transforms == 'torchvision':
        # START TODO #################
        # achieve the same as above with torchvision transforms
        # compare your own implementation against theirs
        raise NotImplementedError
        # END TODO #################
    else:
        raise ValueError(f"Unknown transform {args.transforms}")

    val_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return train_transforms, val_transforms


def train_one_epoch(model, train_loader, epoch, loss_fn, optimizer, args, device):
    print(f"---------- Start of epoch {epoch + 1}")
    # iterate over the training set
    model.train()
    total_loss, total_acc, num_batches = 0., 0., 0
    for batch_idx, (data, label) in enumerate(train_loader):
        num_batches += 1
        # move data to our device
        data = data.to(device)
        label = label.to(device)

        # training process is as follows:
        # 1) use optimizer.zero_grad() to zero the gradients in the optimizer
        # 2) compute the output of the model given the data by using model(input)
        # 3) compute the loss between the output and the label by using loss_fn(output, label)
        # 4) use loss.backward() to accumulate the gradients
        # 5) use optimizer.step() to update the weights
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        predictions = th.argmax(output, dim=1)
        correct = th.sum(predictions == label)
        acc = correct / len(data)

        total_loss += loss.item()
        total_acc += acc.item()
        # log the loss
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs} step {batch_idx}/{len(train_loader)} "
                  f"loss {loss.item():.6f}")
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return avg_loss, avg_acc


def evaluate_one_epoch(model, test_loader, epoch, device, loss_fn, args):
    print(f"---------- Evaluate epoch {epoch + 1}")
    model.eval()
    with torch.no_grad():
        total_loss, total_acc, num_batches = 0., 0., 0
        for batch_idx, (data, label) in enumerate(test_loader):
            num_batches += 1
            # move data to our device
            data = data.to(device)
            label = label.to(device)

            # 1) compute the output of the model given the data
            # 2) compute the loss between the output and the label
            # 3) predictions are given as logits, i.e. pre-softmax class probabilities.
            #     use th.argmax over axis 1 of the logits to get the predictions
            # 4) compute the accuracy by comparing the predictions with the labels
            #   - use predictions == labels to get the correctness for each prediction
            #   - use th.sum to get the total number of correct predictions
            #   - divide by the batchsize to get the accuracy
            output = model(data)
            loss = loss_fn(output, label)
            predictions = th.argmax(output, dim=1)
            correct = th.sum(predictions == label)
            acc = correct / len(data)

            total_loss += loss.item()
            total_acc += acc.item()

            # log the loss
            if batch_idx % 100 == 0:
                print(f"Validation of epoch {epoch}/{args.num_epochs} "
                      f"step {batch_idx}/{len(test_loader)}")
        # normalize accuracy and loss over the number of batches
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser("Deep Learning on CIFAR10")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Default learning rate")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Which optimizer to use")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of parallel workers")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num_filters", type=int, default=32,
                        help="Number of convolutional filters")
    parser.add_argument("--test_dataloader", action="store_true", help="Test the dataloader")
    parser.add_argument("--test_model", action="store_true", help="Test model")
    parser.add_argument("--validate_only", action="store_true", help="Do not run the training loop")
    parser.add_argument("--load_model", type=str, default="", help="Model file to load")
    parser.add_argument('--out_dir', type=str, default='./out', help='path to cifar')
    parser.add_argument('--transforms', type=str, default='basic',
                        help='which transformations to use',
                        choices=['basic', 'own', 'torchvision'])
    parser.add_argument('--subsample_factor', type=float, default=0.2,
                        help='decreases dataset size to speed up overfitting. '
                             'Keep fixed for exercise. You can play with this in the end')
    parser.add_argument('--datafolder', type=str, default='./data',
                        help='path to cifar')
    args = parser.parse_args()

    # START TODO #################
    # initialize the SummaryWriter with a log directory in args.out_dir/logs
    # tb_writer = 
    raise NotImplementedError
    # END TODO #################

    # START TODO #################
    # get the transforms and pass them to the dataset
    # train_transforms, val_transforms = ...
    raise NotImplementedError
    # END TODO #################

    # create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args, train_transforms, val_transforms)

    if args.test_dataloader:
        # test the dataloader and exit
        for batch_idx, (data, label) in enumerate(train_loader):
            print(f"Batch {batch_idx}, data {data.shape} {data.dtype}, "
                  f"labels {label.shape} {label.dtype}")
            if batch_idx == 3:
                break
            print(f"Test of dataloader complete.")
        return

    # set our device
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    print(f"Running training on: {device}")

    # create model
    model = LeNet()

    if args.load_model != "":
        print(f"Load model weights from {args.load_model}")
        # load model weights from the file given by args.load_model and apply them to the model
        weights = th.load(args.load_model)
        model.load_state_dict(weights)

    # move the model to our device
    model = model.to(device)

    if args.test_model:
        # test the model with random noise data and exit
        random_input = th.randn((args.batch_size, 3, 32, 32), dtype=th.float32)
        random_input = random_input.to(device)
        output = model(random_input)
        print(f"Model test complete. Output: {output.shape}")
        return

    # Create the loss function (nn.CrossEntropyLoss)
    loss_fn = nn.CrossEntropyLoss()

    # create optimizer given the string in args.optimizer
    if args.optimizer == "sgd":
        # create stochastic gradient descent optimizer (optim.SGD)
        # given model.parameters() and args.learning_rate
        optimizer = optim.SGD(model.parameters(), args.learning_rate)
    elif args.optimizer == "adamw":
        # create AdamW optimizer (optim.AdamW) given model.parameters() and args.learning_rate
        optimizer = optim.AdamW(model.parameters(), args.learning_rate)
    else:
        raise ValueError(f"Undefined optimizer: {args.optimizer}")

    max_epochs = args.num_epochs
    do_train = True
    if args.validate_only:
        # given this flag we don't want to train but only evaluate the model
        max_epochs = 1
        do_train = False

    # now we run the optimization loop
    epoch = -1
    for epoch in range(max_epochs):
        if do_train:
            train_loss, train_acc = train_one_epoch(model, train_loader, epoch, loss_fn, optimizer,
                                                    args, device)
            # START TODO ###################
            # add_scalar train_loss and train_acc to tb_writer
            raise NotImplementedError
            # END TODO ###################

        # iterate over the val set to compute the accuracy
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, epoch, device, loss_fn, args)
        print(f"Validation of epoch {epoch + 1} complete. "
              f"Loss {val_loss:.6f} accuracy {val_acc:.2%}")
        # START TODO #################
        # add_scalar val_loss and val_acc to tb_writer
        raise NotImplementedError
        # END TODO ###################
        print(f"---------- End of epoch {epoch + 1}")

    model_file = (f"model_e{args.num_epochs}_{args.optimizer}_f{args.num_filters}_"
                  f"lr{args.learning_rate:.1e}.pth")
    model_file = os.path.join(args.out_dir, model_file)
    print(f"Saving model to {model_file}")
    # save the model to disk
    th.save(model.state_dict(), model_file)

    # test the model that has been trained
    test_loss, test_acc = evaluate_one_epoch(model, test_loader, epoch, device, loss_fn, args)
    print(f"Test complete. Loss: {test_loss:.6f} accuracy {test_acc:.2%}")
    # START TODO #################
    # add_scalar test_loss and test_acc to tb_writer
    # ideally, you'd remember the model with the best validation performance and test on that
    raise NotImplementedError
    # END TODO ###################


if __name__ == '__main__':
    main()
    # START TODO ###################
    # implement all TODO's in the script above
    # train the network for 256 epochs
    # use the flag --transforms basic, and specify out_dir as 'no_augment'
    # --- do not put code here ---
    raise NotImplementedError
    # END TODO ###################

    # START TODO ###################
    # train the network a second time with the flag --transforms own and the out_dir 'augment'
    # --- do not put code here ---
    raise NotImplementedError
    # END TODO ###################

    # START TODO ###################
    # visualize the logs with tensorboard
    # do the following for the two networks that we have trained above:
    # (1) run the tensorboard from a commandline from the login (login.informatik.uni-freiburg.de)
    #   - provide it with the path to the logdir
    #   - specify a port --port from the range 51000-55000, they should be free
    # (2) connect via ssh to the login using (replace 6006 with the port number you specified):
    #   (2.1) ssh -L 16006:127.0.0.1:6006 account@server.address
    #         and then refer to http://127.0.0.1:6006/ in a local browser.
    #   or
    #   (2.2) spot the vs-code popup that says the application is now running and
    #         press 'open-in-local-browser'. if it does not pop up: hover over link in cmd
    #         and click on 'follow link using forwarded port'
    # (3) compare the training and validation curves of 'no_augment' and 'augment'
    #   see also the test performance
    # --- do not put code here ---
    raise NotImplementedError
    # END TODO ###################
