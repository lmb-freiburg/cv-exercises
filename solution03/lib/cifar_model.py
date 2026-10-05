"""Model for CIFAR10."""

import torch as th
from torch import nn


class ConvModel(nn.Module):
    def __init__(self, input_channels: int, num_filters: int, verbose: bool = False):
        """
        Model definition.

        Args:
            input_channels: Number of input channels, this is 3 for the RGB images in CIFAR10
            num_filters: Number of convolutional filters
        """
        super().__init__()
        self.verbose = verbose
        self.num_filters = num_filters
        # first convolutional layer

        # Time to define our network. Hints:
        #   Use nn.Conv2d for convolutional layers and set padding="same"
        #   to automatically pad the input to the correct size.
        #   Make each module (i.e. component of the network) an attribute of the class, i.e.
        #   self.conv1 = nn.Conv2d(...)
        #   This is necessary for PyTorch to work correctly.
        # Define the network as follows:
        # 1) Convolution layer with input_channels, output_channels as num_filters,
        #     kernel size 3, stride 2, padding 1, followed by batch norm (nn.BatchNorm2d) and relu (nn.ReLU).
        # 2) Another conv layer with input_channels as num_filters, output_channels as 2 * num_filters,
        #     kernel_size 3, stride 1, padding 1, followed by another batch norm and relu.
        # 3) Averagepooling (nn.AvgPool2d) with kernel size 16, stride 16.
        # 4) Linear layer (nn.Linear) with input_features=2 * num_filters, output_features=10.
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(num_filters, 2 * num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters * 2)
        self.relu2 = nn.ReLU()

        self.avgpool = nn.AvgPool2d(kernel_size=16, stride=16)

        self.linear = nn.Linear(2 * num_filters, 10)

    def forward(self, x: th.Tensor):
        """
        Model forward pass.

        Args:
            x: Model input, shape [batch_size, in_c, in_h, in_w]

        Returns:
            Model output, shape [batch_size, num_classes]
        """
        if self.verbose:
            print(f"Input shape: {x.shape}")
        # Apply first convolutional layer, batch norm and relu.
        # x = self.conv1(x)
        # ...
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if self.verbose:
            print(f"Shape after first layer: {x.shape}")
        # Apply second convolutional layer, batch norm and relu
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.verbose:
            print(f"Shape after second layer: {x.shape}")
        # Apply averagepool
        x = self.avgpool(x)
        if self.verbose:
            print(f"Shape after averagepool: {x.shape}")

        # Here we reshape the input to 2D shape (batch_size, 2 * num_filters)
        # so we can apply a linear layer.
        x = th.reshape(x, (-1, 2 * self.num_filters))
        if self.verbose:
            print(f"Shape after reshape: {x.shape}")

        # Apply the linear.
        x = self.linear(x)
        if self.verbose:
            print(f"Model output shape: {x.shape}")
        return x
