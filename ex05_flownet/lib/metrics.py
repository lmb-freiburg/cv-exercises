import torch
import torch.nn.functional as F


def aepe(gt, pred, weight=None):
    """Computes the Average Endpoint Error (EPE) between a ground truth
    and predicted displacement field.

     The displacement field can be an optical flow (2d) or disparity field (1d).
     The Average Endpoint Error is the Euclidean distance between ground truth
     and predicted displacements, averaged over all pixels.

    Args:
        gt: Ground truth displacement field with shape (N,C,H,W) where C is typically 1 or 2.
        pred: Predicted displacement field with shape (N,C,H,W) where C is typically 1 or 2.
        weight: Optional scalar weight to scale the Average EPE,
            e.g. for weighting the loss during training.

    Returns:
        The Average Endpoint Error (EPE) between the ground truth and prediction.
    """
    # START TODO #################
    # Hint: Check torch.linalg.norm() to compute the norm of a vector.
    raise NotImplementedError
    # END TODO ###################
    return aepe_out


def pointwise_epe(gt, pred, weight=None):
    """Computes the pointwise Endpoint Error (EPE) between a ground truth and
    predicted displacement field.

     The displacement field can be an optical flow (2d) or disparity field (1d).
     The Endpoint Error is the Euclidean distance between ground truth and predicted displacements.

    Args:
        gt: Ground truth displacement field with shape (N,C,H,W) where C is typically 1 or 2.
        pred: Predicted displacement field with shape (N,C,H,W) where C is typically 1 or 2.
        weight: Scalar weight to scale the EPE, e.g. for weighting the loss during training.

    Returns:
        The pointwise Endpoint Error (EPE) between the ground truth and
        the prediction, with shape (N,1,H,W).
    """
    # START TODO #################
    raise NotImplementedError
    # END TODO ###################


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

    aepe_ = aepe(gt=gt_flow, pred=pred_flow).item()
    pointwise_epe_ = pointwise_epe(gt=gt_flow, pred=pred_flow)

    metrics = {
        "aepe": aepe_,
    }

    qualitatives = {
        "pred_flow": pred_flow,
        "epe": pointwise_epe_,
    }
    return metrics, qualitatives
