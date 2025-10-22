# %%
# imports

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable


# %%
# utils


def pad_image(input_image, pad):
    # pad image with mirrored values to avoid boundary effects and index errors
    return np.pad(input_image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")


def print_info(arr):
    print(f"Got array of shape {arr.shape} {arr.dtype} bounds {np.min(arr)} {np.max(arr)}")


def load_image_as_numpy_array(input_file):
    img = Image.open(input_file)
    arr = np.array(img).astype(np.float32)
    print_info(arr)
    return arr


def show_image(img):
    img = np.round(img).astype(int)
    plt.imshow(img)


def add_colorbar(fig, fig_imshow, ax):
    """Add colorbar to a given figure"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(fig_imshow, cax=cax)


# %%
# load image

input_file = "diffusion.png"
in_arr = load_image_as_numpy_array(input_file)
arr = np.copy(in_arr)

show_image(arr)


# %%
# homogeneous diffusion


def apply_homogeneous_diffusion_step(input_image, tau, h):
    # pad image to avoid index errors
    padded_image = pad_image(input_image, 1)

    # START TODO #################
    # apply the explicit finite difference scheme
    # new_image = ...
    raise NotImplementedError
    # END TODO ###################
    return new_image


tau, h, n_steps = 0.1, 1, 10
print(
    f"Running homogeneous diffusion with hyperparameters "
    f"tau={tau} h={h} n_steps={n_steps}"
)
arr_hg = np.copy(in_arr)
for n in range(10):
    arr_hg = apply_homogeneous_diffusion_step(arr_hg, 0.1, 1)

plt.figure(figsize=(16, 8))
both = np.concatenate((in_arr, arr_hg), axis=1)
show_image(both)


# %%
# nonlinear diffusion


def apply_nonlinear_diffusion_step(input_image, tau, lambd, debug=False):
    # pad image to avoid index errors
    padded_image = pad_image(input_image, 1)

    # START TODO #################
    # apply nonlinear isotropic diffusion with gaussian diffusivity
    # 1. compute diffusivity
    # diffu = ...
    raise NotImplementedError
    # END TODO ###################

    if debug:
        print("Diffusivity:")
        print_info(diffu)
        plt.imshow(diffu)
        plt.show()

    # pad diffusivity to avoid index errors
    padded_diffu = pad_image(diffu, 1)

    # 2. compute diffusivities between pixels
    diff_x_plus_half = 0.5 * (padded_diffu[1:-1, 2:] + diffu)
    diff_x_minus_half = 0.5 * (padded_diffu[1:-1, :-2] + diffu)
    diff_y_plus_half = 0.5 * (padded_diffu[2:, 1:-1] + diffu)
    diff_y_minus_half = 0.5 * (padded_diffu[:-2, 1:-1] + diffu)

    # START TODO #################
    # 3. implement the explicit scheme
    # new_image = ...
    raise NotImplementedError
    # END TODO ###################

    if debug:
        print("Difference in current step:")
        delta = np.abs(input_image - new_image)
        print_info(delta)
        fig, ax = plt.subplots()
        fig_imshow = plt.imshow(delta.mean(-1))
        add_colorbar(fig, fig_imshow, ax)
        plt.show()

        fig, ax = plt.subplots()
        show_image(new_image)
        plt.show()

    return new_image


arr = np.copy(in_arr)
out_arr = apply_nonlinear_diffusion_step(arr, 0.1, 10, debug=True)


def apply_nonlinear_diffusion(in_arr, tau, lambd, n_steps):
    print(
        f"Running nonlinear diffusion with hyperparameters "
        f"tau={tau} lambda={lambd} n_steps={n_steps}"
    )
    arr_nid = np.copy(in_arr)
    for n in range(n_steps):
        arr_nid = apply_nonlinear_diffusion_step(arr_nid, tau, lambd)
    plt.figure(figsize=(16, 8))
    both = np.concatenate((in_arr, arr_nid), axis=1)
    show_image(both)
    plt.show()


tau = 0.1
lambd = 10
T = 10
n_steps = np.ceil(T / tau).astype(int)

apply_nonlinear_diffusion(in_arr, tau, lambd, n_steps)
