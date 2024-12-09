import torch
import torch.nn.functional as F

from lib.utils import warp, shift_multi


def epe(gt, pred, weight=None):
    """Computes the Average Endpoint Error (EPE) between a ground truth and predicted displacement field.

     The displacement field can be an optical flow (2d) or disparity field (1d).
     The Average Endpoint Error is the Euclidean distance between ground truth and predicted displacements,
     averaged over all pixels.

    Args:
        gt: Ground truth displacement field with shape (N,C,H,W) where C is typically 1 or 2.
        pred: Predicted displacement field with shape (N,C,H,W) where C is typically 1 or 2.
        weight: Scalar weight to scale the Average EPE, e.g. for weighting the loss during training.

    Returns:
        The Average Endpoint Error (EPE) between the ground truth and prediction.
    """
    epe_ = torch.linalg.norm(pred - gt, dim=1)
    epe_ = epe_ * weight if weight is not None else epe_
    avg_epe = epe_.mean()
    return avg_epe


def pointwise_epe(gt, pred, weight=None):
    """Computes the pointwise Endpoint Error (EPE) between a ground truth and predicted displacement field.

     The displacement field can be an optical flow (2d) or disparity field (1d).
     The Endpoint Error is the Euclidean distance between ground truth and predicted displacements.

    Args:
        gt: Ground truth displacement field with shape (N,C,H,W) where C is typically 1 or 2.
        pred: Predicted displacement field with shape (N,C,H,W) where C is typically 1 or 2.
        weight: Scalar weight to scale the EPE, e.g. for weighting the loss during training.

    Returns:
        The pointwise Endpoint Error (EPE) between the ground truth and prediction with shape (N,1,H,W).
    """
    pointwise_epe_ = torch.linalg.norm(pred - gt, dim=1, keepdim=True)
    pointwise_epe_ *= weight if weight is not None else pointwise_epe_
    return pointwise_epe_


def charbonnierLoss(e, alpha=0.5, eps=1e-3):
    """Computes the charbonnier loss (e**2 + eps**2)**alpha."""
    return (e**2 + eps**2) ** alpha


def photometric_loss(image_1, image_2, pred_flow, weight=None, alpha=0.5, eps=1e-3):
    """Computes the photometric loss between two images given the estimated flow from image 1 to image 2.

    The predicted flow can be of lower resolution than the images. In this case, it will be upsampled using
    bilinear interpolation.
    The loss is computed using the Charbonnier function as distance metric between the original and warped image.

    Args:
        image_1: Image 1 with shape (N,3,H,W).
        image_2: Image 2 with shape (N,3,H,W).
        pred_flow: Predicted flow from image 1 to image 2 with shape (N,2,h,w).
        weight: Scalar weight to scale the photometric loss, e.g. for weighting the loss during training.
        alpha: Alpha value for the Charbonnier function.
        eps: Epsilon value for the Charbonnier function.

    Returns:
        Tuple (image_2_warped, warping_mask, pointwise_photo_loss, photo_loss) containing
            image_2_warped: Image 2 warped with predicted flow with shape (N,3,H,W).
            warping_mask: Mask that indicates where warping was not valid with shape (N,1,H,W).
            pointwise_photo_loss: Photometric loss per pixel with shape (N,1,H,W)
            photo_loss: Photometric loss averaged over all valid pixels (scalar).
    """
    # 1. Scale predicted flow to shape of image_1 using F.interpolate.
    pred_flow = F.interpolate(
        pred_flow, size=image_1.shape[-2:], mode="bilinear", align_corners=False
    )

    # 2. Warp image_2 with the pred_flow using the provided "warp" function.
    image_2_warped, warping_mask = warp(image_2, offset=pred_flow, padding_mode="zeros")

    # 3. Compute the per-pixel photometric error using the Charbonnier function. Mask where warping was invalid.
    # Weight per-pixel error with the given weight if it is not None.
    image_diff = image_1 - image_2_warped
    pointwise_photo_error = (
        torch.sum(charbonnierLoss(image_diff, alpha, eps), dim=1, keepdim=True)
        * warping_mask
    )
    pointwise_photo_loss = (
        pointwise_photo_error * weight if weight is not None else pointwise_photo_error
    )

    # 4. Average the photometric error over all pixels where the warping was valid (using the obtained warping mask).
    warping_mask = warping_mask.float()
    num_valid = torch.sum(warping_mask)
    photo_loss = 1 / (num_valid + 1e-3) * torch.sum(pointwise_photo_loss)
    photo_loss = photo_loss * float((num_valid != 0))

    return image_2_warped, warping_mask, pointwise_photo_loss, photo_loss


def smoothness_loss(pred_flow, weight=None, alpha=0.3, eps=1e-3):
    """Computes the smoothness loss for a predicted flow.

    Args:
        pred_flow: Predicted flow with shape (N,2,H,W).
        weight: Scalar weight to scale the loss, e.g. for weighting the loss during training.
        alpha: Alpha value for the Charbonnier function.
        eps: Epsilon value for the Charbonnier function.

    Returns:
        Tuple (pointwise_smoothness_loss, smoothness_loss) containing
            pointwise_smoothness_loss: Smoothness loss per pixel with shape (N,1,H,W)
            smoothness_loss: Smoothness loss averaged over all valid pixels (scalar).
    """
    # 1. Compute difference of neighbouring (to the right and to bellow) flow values.
    offsets = torch.tensor([[1, 0], [0, 1]])
    shifteds, shift_masks = shift_multi(pred_flow, offsets)  # N22HW, NSHW
    flow_diffs = pred_flow.unsqueeze(1) - shifteds  # N22HW

    # 2. Compute the pointwise smoothness loss by penalizing neighbouring flow differences with the Charbonnier function
    # and summing up the penalties. Mask invalid (out-of-image) neighbours.
    # Weight per-pixel loss with the given weight if it is not None.
    pointwise_smoothness_loss = torch.sum(
        torch.sum(charbonnierLoss(flow_diffs, alpha, eps), dim=2), dim=1, keepdim=True
    )  # N1HW
    mask = torch.sum(shift_masks, dim=1, keepdim=True) == len(offsets)  # N1HW
    pointwise_smoothness_loss = pointwise_smoothness_loss * mask
    pointwise_smoothness_loss = (
        pointwise_smoothness_loss * weight
        if weight is not None
        else pointwise_smoothness_loss
    )

    # 3. Average the smoothness loss over all pixels where the neighbours were valid.
    mask = mask.float()
    num_valid = torch.sum(mask)
    smoothness_loss_ = 1 / (num_valid + eps) * torch.sum(pointwise_smoothness_loss)
    smoothness_loss_ = smoothness_loss_ * float((num_valid != 0))

    return pointwise_smoothness_loss, smoothness_loss_


def compute_flow_metrics(sample, model_output):
    image = sample["images"][0]
    gt_flow = sample["gt_flow"]
    pred_flow = model_output["pred_flow"]

    orig_ht, orig_wd = gt_flow.shape[-2:]
    pred_ht, pred_wd = image.shape[-2:]
    scale_ht, scale_wd = orig_ht / pred_ht, orig_wd / pred_wd

    pred_flow = F.interpolate(pred_flow, size=gt_flow.shape[-2:], mode="nearest")
    pred_flow[:, 0, :, :] = pred_flow[:, 0, :, :] * scale_wd
    pred_flow[:, 1, :, :] = pred_flow[:, 1, :, :] * scale_ht

    epe_ = epe(gt=gt_flow, pred=pred_flow).item()
    pointwise_epe_ = pointwise_epe(gt=gt_flow, pred=pred_flow)

    metrics = {
        "epe": epe_,
    }

    qualitatives = {
        "pred_flow": pred_flow,
        "epe": pointwise_epe_,
    }
    return metrics, qualitatives


def compute_disp_metrics(sample, model_output):
    # note: either detach() or torch.no_grad() is needed to avoid memory leaks
    image = sample["images"][0].detach()
    gt_disp = sample["gt_disp"].detach()

    if "pred_disp" in model_output:
        pred_disp = model_output["pred_disp"].detach()
    else:
        pred_disp = model_output["pred_flow"][:, 0:1, :, :].detach()

    orig_ht, orig_wd = gt_disp.shape[-2:]
    pred_ht, pred_wd = image.shape[-2:]
    scale_ht, scale_wd = orig_ht / pred_ht, orig_wd / pred_wd

    pred_disp = F.interpolate(pred_disp, size=gt_disp.shape[-2:], mode="nearest")
    pred_disp = pred_disp * scale_wd

    epe_ = epe(gt=gt_disp, pred=pred_disp).item()
    pointwise_epe_ = pointwise_epe(gt=gt_disp, pred=pred_disp)

    metrics = {
        "epe": epe_,
    }

    qualitatives = {
        "pred_disp": pred_disp,
        "epe": pointwise_epe_,
    }
    return metrics, qualitatives
