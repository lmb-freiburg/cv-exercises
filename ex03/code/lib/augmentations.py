import torch
from PIL import Image
import numpy as np


class horizontal_flip(torch.nn.Module):
    """
    Flip the image along the second dimension with a probability of p
    """

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, img):
        # START TODO #################
        # convert the image to numpy
        # draw a random number
        # flip the image in the second dimension 
        # if this number is smaller than self.p
        raise NotImplementedError
        # END TODO #################


class random_resize_crop(torch.nn.Module):
    """
    simplified version of resize crop, which keeps the aspect ratio of the image.
    """

    def __init__(self, size, scale):
        """ initialize this transform
        Args:
            size (int): size of the image
            scale (tuple(int)): upper and lower bound for resizing image"""
        super().__init__()
        self.size = size
        self.scale = scale

    def _uniform_rand(self, low, high):
        return np.random.rand(1)[0] * (low - high) + high

    def forward(self, img):
        # START TODO #################
        raise NotImplementedError
        # END TODO #################
