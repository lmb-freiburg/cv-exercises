import torch
import torchvision
import numpy as np
from torch.utils.tensorboard import summary

from lib.vis import np2d, np3d


def log_histogram(writer, desc, tensor, step, idx=None, replace_NaNs=False):
    desc = desc + "/" + str(idx) if (idx is not None and isinstance(idx, int)) else desc
    desc = desc + idx if (idx is not None and isinstance(idx, str)) else desc

    if replace_NaNs:
        tensor = summary.make_np(tensor).copy()
        tensor[~np.isfinite(tensor)] = 0

    writer.add_histogram(desc, tensor, step)


def log_scalar(writer, desc, tensor, step):
    writer.add_scalar(desc, tensor, step)


def log_tensor(
    writer,
    desc,
    tensor,
    step,
    idx=None,
    full_batch=False,
    colorize=True,
    clipping=False,
    upper_clipping_thresh=None,
    lower_clipping_thresh=None,
    mark_clipping=False,
    clipping_color=None,
    invalid_values=None,
    mark_invalid=False,
    invalid_color=None,
    text=None,
    cmap=None,
    markers=None,
    marker_radius=None,
    marker_text_off=False,
    marker_cmap=None,
    ignore_marker_scores=False,
    min_marker_score=None,
    max_marker_score=None,
    image_range_text_off=False,
    marker_range_text_off=False,
    text_off=False,
):
    desc = desc + "/" + str(idx) if (idx is not None and isinstance(idx, int)) else desc
    desc = desc + idx if (idx is not None and isinstance(idx, str)) else desc

    if full_batch:
        tensor = torch.from_numpy(tensor) if isinstance(tensor, np.ndarray) else tensor
        tensor = torch.unsqueeze(torchvision.utils.make_grid(tensor)[0], 0)
    else:
        tensor = tensor[0]

    tensor = summary.make_np(tensor)

    ch = tensor.shape[0]
    if ch == 1:
        img = np2d(
            tensor,
            colorize=colorize,
            clipping=clipping,
            upper_clipping_thresh=upper_clipping_thresh,
            lower_clipping_thresh=lower_clipping_thresh,
            mark_clipping=mark_clipping,
            clipping_color=clipping_color,
            invalid_values=invalid_values,
            mark_invalid=mark_invalid,
            invalid_color=invalid_color,
            text=text,
            cmap=cmap,
            markers=markers,
            marker_radius=marker_radius,
            marker_text_off=marker_text_off,
            marker_cmap=marker_cmap,
            ignore_marker_scores=ignore_marker_scores,
            min_marker_score=min_marker_score,
            max_marker_score=max_marker_score,
            image_range_text_off=image_range_text_off,
            marker_range_text_off=marker_range_text_off,
            text_off=text_off,
            out_format={
                "type": "np",
                "mode": "RGB",
                "channels": "CHW",
                "dtype": "uint8",
            },
        )

    elif ch == 2:
        img = np3d(
            arr=tensor,
            channels="FLOW",
            text=text,
            gray=(not colorize),
            clipping=clipping,
            upper_clipping_thresh=upper_clipping_thresh,
            lower_clipping_thresh=lower_clipping_thresh,
            mark_clipping=mark_clipping,
            clipping_color=clipping_color,
            invalid_values=invalid_values,
            mark_invalid=mark_invalid,
            invalid_color=invalid_color,
            markers=markers,
            marker_radius=marker_radius,
            marker_text_off=marker_text_off,
            marker_cmap=marker_cmap,
            ignore_marker_scores=ignore_marker_scores,
            min_marker_score=min_marker_score,
            max_marker_score=max_marker_score,
            image_range_text_off=image_range_text_off,
            marker_range_text_off=marker_range_text_off,
            text_off=text_off,
            out_format={
                "type": "np",
                "mode": "RGB",
                "channels": "CHW",
                "dtype": "uint8",
            },
        )

    elif ch == 3:
        img = np3d(
            arr=tensor,
            channels="RGB",
            text=text,
            gray=(not colorize),
            clipping=clipping,
            upper_clipping_thresh=upper_clipping_thresh,
            lower_clipping_thresh=lower_clipping_thresh,
            mark_clipping=mark_clipping,
            clipping_color=clipping_color,
            invalid_values=invalid_values,
            mark_invalid=mark_invalid,
            invalid_color=invalid_color,
            markers=markers,
            marker_radius=marker_radius,
            marker_text_off=marker_text_off,
            marker_cmap=marker_cmap,
            ignore_marker_scores=ignore_marker_scores,
            min_marker_score=min_marker_score,
            max_marker_score=max_marker_score,
            image_range_text_off=image_range_text_off,
            marker_range_text_off=marker_range_text_off,
            text_off=text_off,
            out_format={
                "type": "np",
                "mode": "RGB",
                "channels": "CHW",
                "dtype": "uint8",
            },
        )

    else:
        raise ValueError(
            f"log_tensor was called with a tensor of shape {tensor.shape} but "
            "can only be used with a NCHW tensor with C=1, 2 or 3."
        )

    writer.add_image(desc, img, step)


def log_tensor_list(
    writer,
    desc,
    tensor_list,
    step,
    labels=None,
    idx=None,
    full_batch=False,
    every_nth=None,
    colorize=True,
    clipping=False,
    upper_clipping_thresh=None,
    lower_clipping_thresh=None,
    mark_clipping=False,
    clipping_color=None,
    invalid_values=None,
    mark_invalid=False,
    invalid_color=None,
    text=None,
    cmap=None,
    markers=None,
    marker_radius=None,
    marker_text_off=False,
    marker_cmap=None,
    ignore_marker_scores=False,
    min_marker_score=None,
    max_marker_score=None,
    image_range_text_off=False,
    marker_range_text_off=False,
    text_off=False,
):
    for tensor_idx, tensor in enumerate(tensor_list):
        if every_nth is not None and tensor_idx % every_nth != 0:
            continue

        idx_label = (
            "/" + str(idx) if (idx is not None and not isinstance(idx, str)) else None
        )
        idx_label = idx if (idx is not None and isinstance(idx, str)) else idx_label

        label = tensor_idx if labels is None else labels[tensor_idx]
        label = (
            (
                idx_label + "/" + str(label)
                if not isinstance(label, str)
                else idx_label + label
            )
            if idx_label is not None
            else label
        )

        log_tensor(
            writer=writer,
            desc=desc,
            tensor=tensor,
            step=step,
            idx=label,
            full_batch=full_batch,
            colorize=colorize,
            clipping=clipping,
            upper_clipping_thresh=upper_clipping_thresh,
            lower_clipping_thresh=lower_clipping_thresh,
            mark_clipping=mark_clipping,
            clipping_color=clipping_color,
            invalid_values=invalid_values,
            mark_invalid=mark_invalid,
            invalid_color=invalid_color,
            text=text,
            cmap=cmap,
            markers=markers,
            marker_radius=marker_radius,
            marker_text_off=marker_text_off,
            marker_cmap=marker_cmap,
            ignore_marker_scores=ignore_marker_scores,
            min_marker_score=min_marker_score,
            max_marker_score=max_marker_score,
            image_range_text_off=image_range_text_off,
            marker_range_text_off=marker_range_text_off,
            text_off=text_off,
        )


class Logger:
    def __init__(self, writer, model, optimizer=None):
        self.writer = writer
        self.model = model
        self.optimizer = optimizer
        self.overview_base = "0_overview"
        self.log_full_batch = False

    def log(
        self,
        sample,
        model_output,
        loss,
        sub_losses,
        pointwise_losses,
        step,
        loss_only=False,
    ):
        if not loss_only:
            self.log_in_data(sample, step)
            self.log_pred(model_output, step)
            self.log_loss(loss, sub_losses, pointwise_losses, step, scalars_only=False)
            self.log_optim(step)
        else:
            self.log_loss(loss, sub_losses, pointwise_losses, step, scalars_only=True)

    def log_eval(
        self, sample, model_output, sample_idx, metrics=None, qualitatives=None
    ):
        self.log_in_data(sample, step=sample_idx)
        self.log_pred(model_output, step=sample_idx)
        self.log_metrics(metrics, qualitatives, step=sample_idx)

    # old fn
    def log_in_data(self, sample, step):
        base_name = "1_in"
        full_batch = self.log_full_batch

        images = sample["images"]
        gt_flow = sample["gt_flow"] if "gt_flow" in sample else None
        gt_disp = sample["gt_disp"] if "gt_disp" in sample else None

        log_tensor(
            self.writer,
            "{}/{}/00_images/ref".format(self.overview_base, base_name),
            images[0],
            step,
            full_batch=full_batch,
        )
        log_tensor_list(
            self.writer, base_name + "/00_images", images, step, full_batch=full_batch
        )

        if gt_flow is not None:
            log_tensor(
                self.writer,
                base_name + "/01_gt_flow",
                gt_flow,
                step,
                full_batch=full_batch,
            )
            log_tensor(
                self.writer,
                "{}/{}/01_gt_flow".format(self.overview_base, base_name),
                gt_flow,
                step,
                full_batch=full_batch,
            )

        if gt_disp is not None:
            log_tensor(
                self.writer,
                base_name + "/02_gt_disp",
                gt_disp,
                step,
                full_batch=full_batch,
            )
            log_tensor(
                self.writer,
                "{}/{}/02_gt_disp".format(self.overview_base, base_name),
                gt_disp,
                step,
                full_batch=full_batch,
            )

    def log_model(self, sample, step):
        base_name = "2_model"
        full_batch = self.log_full_batch

        model_vis = self.model.visualize(sample)

        for model_vis_name, vis in model_vis.items():
            log_tensor(
                self.writer,
                "{}/{}".format(base_name, model_vis_name),
                vis,
                step,
                full_batch=full_batch,
            )

    def log_pred(self, model_output, step):
        base_name = "3_pred"
        full_batch = self.log_full_batch

        pred_flow = model_output["pred_flow"] if "pred_flow" in model_output else None
        pred_flows_all = (
            model_output["pred_flows_all"] if "pred_flows_all" in model_output else None
        )

        pred_disp = model_output["pred_disp"] if "pred_disp" in model_output else None
        pred_disps_all = (
            model_output["pred_disps_all"] if "pred_disps_all" in model_output else None
        )

        if pred_flow is not None:
            log_tensor(
                self.writer,
                base_name + "/01a_pred_flow",
                pred_flow,
                step,
                full_batch=full_batch,
            )
            log_tensor(
                self.writer,
                "{}/{}/01_pred_flow".format(self.overview_base, base_name),
                pred_flow,
                step,
                full_batch=full_batch,
            )

        if pred_flows_all is not None:
            for level, pred_flow_ in enumerate(pred_flows_all):
                log_tensor(
                    self.writer,
                    base_name + "/01b_pred_flows_all/level_%d" % level,
                    pred_flow_,
                    step,
                    full_batch=full_batch,
                )

        if pred_disp is not None:
            log_tensor(
                self.writer,
                base_name + "/02a_pred_disp",
                pred_disp,
                step,
                full_batch=full_batch,
            )
            log_tensor(
                self.writer,
                "{}/{}/02_pred_disp".format(self.overview_base, base_name),
                pred_disp,
                step,
                full_batch=full_batch,
            )

        if pred_disps_all is not None:
            for level, pred_disp_ in enumerate(pred_disps_all):
                log_tensor(
                    self.writer,
                    base_name + "/02b_pred_disps_all/level_%d" % level,
                    pred_disp_,
                    step,
                    full_batch=full_batch,
                )

    def log_loss(self, loss, sub_losses, pointwise_losses, step, scalars_only):
        base_name = "4_loss"
        full_batch = self.log_full_batch

        log_scalar(self.writer, base_name + "/0_total_loss", loss, step)
        log_scalar(
            self.writer,
            "{}/{}/0_total_loss".format(self.overview_base, base_name),
            loss,
            step,
        )

        for sub_name, sub_val in sub_losses.items():
            log_scalar(
                self.writer, "{}/1_sub/{}".format(base_name, sub_name), sub_val, step
            )

        if not scalars_only:
            for pointwise_name, pointwise_loss in pointwise_losses.items():
                log_tensor(
                    self.writer,
                    "{}/{}".format(base_name, pointwise_name),
                    pointwise_loss,
                    step,
                    full_batch=full_batch,
                )

    def log_optim(self, step):
        base_name = "5_optim"
        optimizer = self.optimizer
        model = self.model

        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group["lr"]
            log_scalar(
                self.writer,
                "{}/{}/lr/group_{}".format(self.overview_base, base_name, i),
                lr,
                step,
            )

        for name, param in model.named_parameters():
            log_histogram(
                self.writer, "{}/0_vals/{}".format(base_name, name), param, step
            )

            if param.grad is not None:
                log_histogram(
                    self.writer,
                    "{}/1_grads/{}".format(base_name, name),
                    param.grad,
                    step,
                    replace_NaNs=False,
                )

    def log_metrics(self, metrics, qualitatives, step):
        base_name = "6_metrics"
        full_batch = self.log_full_batch

        for idx, (metric_name, metric_val) in enumerate(metrics.items()):
            log_scalar(
                self.writer,
                "{}/{}_{}".format(base_name, idx, metric_name),
                metric_val,
                step,
            )

        for idx, (qual_name, qual) in enumerate(qualitatives.items()):
            log_tensor(
                self.writer,
                "{}/{}_{}".format(base_name, idx, qual_name),
                qual,
                step,
                full_batch=full_batch,
            )
