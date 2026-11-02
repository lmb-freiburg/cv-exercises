"""Simple example script."""

import argparse

import torch as th
from torch import nn, optim

from lib.cifar_dataset import create_cifar_datasets, create_dataloader
from lib.cifar_model import ConvModel


def main():
    parser = argparse.ArgumentParser("Deep Learning on CIFAR10")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
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
    args = parser.parse_args()

    # create datasets and dataloaders
    train_set, test_set = create_cifar_datasets()
    train_loader = create_dataloader(train_set, args.batch_size, is_train=True,
                                     num_workers=args.num_workers)
    test_loader = create_dataloader(test_set, args.batch_size, is_train=False,
                                    num_workers=args.num_workers)

    if args.test_dataloader:
        # test the dataloader and exit
        for batch_idx, (data, label) in enumerate(train_loader):
            print(
                    f"Batch {batch_idx}, data {data.shape} {data.dtype}, labels {label.shape} {label.dtype}")
            if batch_idx == 3:
                break
            print(f"Test of dataloader complete.")
        return

    # set our device
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    print(f"Running training on: {device}")

    # create model with 3 input channels for the RGB images
    model = ConvModel(3, args.num_filters, verbose=args.test_model)

    if args.load_model != "":
        print(f"Load model weights from {args.load_model}")
        # START TODO #################
        # load model weights from the file given by args.load_model and apply them to the model
        # weights = th.load ...
        # model.load_state_dict ...
        raise NotImplementedError
        # END TODO ###################

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
    # START TODO #################
    # loss_fn = ...
    raise NotImplementedError
    # END TODO ###################

    # create optimizer given the string in args.optimizer
    if args.optimizer == "sgd":
        # START TODO #################
        # create stochastic gradient descent optimizer (optim.SGD) given model.parameters() and args.learning_rate
        # optimizer = ...
        raise NotImplementedError
        # END TODO ###################
    elif args.optimizer == "adamw":
        # START TODO #################
        # create AdamW optimizer (optim.AdamW) given model.parameters() and args.learning_rate
        # optimizer = ...
        raise NotImplementedError
        # END TODO ###################
    else:
        raise ValueError(f"Undefined optimizer: {args.optimizer}")

    max_epochs = args.num_epochs
    do_train = True
    if args.validate_only:
        # given this flag we don't want to train but only evaluate the model
        max_epochs = 1
        do_train = False

    # now we run the optimization loop
    for epoch in range(max_epochs):
        if do_train:
            print(f"---------- Start of epoch {epoch + 1}")
            # iterate over the training set
            model.train()
            for batch_idx, (data, label) in enumerate(train_loader):
                # move data to our device
                data = data.to(device)
                label = label.to(device)

                # START TODO #################
                # training process is as follows:
                # 1) use optimizer.zero_grad() to zero the gradients in the optimizer
                # 2) compute the output of the model given the data by using model(input)
                # 3) compute the loss between the output and the label by using loss_fn(output, label)
                # 4) use loss.backward() to accumulate the gradients
                # 5) use optimizer.step() to update the weights
                raise NotImplementedError
                # END TODO ###################

                # log the loss
                if batch_idx % 100 == 0:
                    print(
                            f"Epoch {epoch + 1}/{args.num_epochs} step {batch_idx}/{len(train_loader)} "
                            f"loss {loss.item():.6f}")

        # iterate over the test set to compute the accuracy
        print(f"---------- Evaluate epoch {epoch + 1}")
        model.eval()
        with th.no_grad():
            total_loss, total_acc, num_batches = 0., 0., 0
            for batch_idx, (data, label) in enumerate(test_loader):
                num_batches += 1
                # move data to our device
                data = data.to(device)
                label = label.to(device)

                # START TODO #################
                # 1) compute the output of the model given the data
                # 2) compute the loss between the output and the label
                # 3) predictions are given as logits, i.e. pre-softmax class probabilities.
                #     use th.argmax over axis 1 of the logits to get the predictions
                # 4) compute the accuracy by comparing the predictions with the labels
                #   - use predictions == labels to get the correctness for each prediction
                #   - use th.sum to get the total number of correct predictions
                #   - divide by the batchsize to get the accuracy
                raise NotImplementedError
                # END TODO ###################

                total_loss += loss.item()
                total_acc += acc.item()

                # log the loss
                if batch_idx % 100 == 0:
                    print(
                            f"Validation of epoch {epoch}/{args.num_epochs} step {batch_idx}/{len(test_loader)}")
        # normalize accuracy and loss over the number of batches
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        print(
                f"Validation of epoch {epoch + 1} complete. Loss {avg_loss:.6f} accuracy {avg_acc:.2%}")

        print(f"---------- End of epoch {epoch + 1}")

    model_file = (f"model_e{args.num_epochs}_{args.optimizer}_f{args.num_filters}_"
                  f"lr{args.learning_rate:.1e}.pth")
    print(f"Saving model to {model_file}")
    # START TODO #################
    # save the model to disk by using th.save with parameters model.state_dict() and model_file
    raise NotImplementedError
    # END TODO ###################


if __name__ == '__main__':
    main()
