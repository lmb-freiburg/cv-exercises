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

        # convert to numpy. This is slower than using PIL but more instructional.
        img = np.array(img)
        if torch.rand(1) < self.p:
            img = img[:, ::-1, :]
        return Image.fromarray(img)
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

        # resize the image using img.resize
        # determine the new size from a random scale between self.scale[0] and self.scale[1]
        scale = self._uniform_rand(self.scale[0], self.scale[1])
        w, h = img.size
        new_size = np.array(np.round((w * scale, h * scale)), dtype=int)
        img = img.resize(new_size, resample=Image.BILINEAR)

        # again we cast to numpy but using PIL would be faster
        img = np.array(img)
        # determine crop indices
        max_top_left = (new_size[0] - w, new_size[1] - h)
        top_left = (self._uniform_rand(0, max_top_left[0]),
                    self._uniform_rand(0, max_top_left[1]))
        bottom_right = (top_left[0] + self.size,
                        top_left[1] + self.size)

        # round and cast indices to int
        top_left = np.array(np.round(top_left), dtype=int)
        bottom_right = np.array(np.round(bottom_right), dtype=int)
        # crop
        crop = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]
        # return the image
        return Image.fromarray(crop)
        # END TODO #################
