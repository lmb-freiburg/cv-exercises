import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.metrics import epe, pointwise_epe, photometric_loss, smoothness_loss


class FlowLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.weight_decay = 1e-4
        self.gt_interpolation = "bilinear"

        self.loss_weights = [1 / 16, 1 / 16, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1]
        self.loss_weights = [100 * weight for weight in self.loss_weights]
        self.reg_params = self.get_regularization_parameters(model)

    def get_regularization_parameters(self, model):
        reg_params = []
        for name, param in model.named_parameters():
            if (
                "pred" not in name
                and not name.endswith("bias")
                and not name.endswith("bn.weight")
                and param.requires_grad
            ):
                reg_params.append((name, param))

        print(
            "Applying regularization loss with weight decay {} on:".format(
                self.weight_decay
            )
        )
        for i, val in enumerate(reg_params):
            name, param = val
            print("\t#{} {}: {} ({})".format(i, name, param.shape, param.numel()))
        print()

        return reg_params

    def forward(self, sample, model_output):
        pointwise_losses = {}
        sub_losses = {}

        gt_flow = sample["gt_flow"]
        pred_flows_all = model_output["pred_flows_all"]

        total_epe_loss = 0
        total_reg_loss = 0

        for level, pred_flow in enumerate(pred_flows_all):
            with torch.no_grad():
                gt_flow_resampled = F.interpolate(
                    gt_flow,
                    size=pred_flow.shape[-2:],
                    mode=self.gt_interpolation,
                    align_corners=(
                        False if self.gt_interpolation != "nearest" else None
                    ),
                )

            epe_loss = epe(
                gt=gt_flow_resampled, pred=pred_flow, weight=self.loss_weights[level]
            )
            pointwise_epe_ = pointwise_epe(
                gt=gt_flow_resampled, pred=pred_flow, weight=self.loss_weights[level]
            )

            sub_losses["0_epe/level_%d" % level] = epe_loss
            pointwise_losses["0_epe/level_%d" % level] = pointwise_epe_

            total_epe_loss += epe_loss

        for name, param in self.reg_params:
            reg_loss = torch.sum(torch.mul(param, param)) / 2.0
            total_reg_loss += reg_loss

        total_reg_loss *= self.weight_decay

        total_loss = total_epe_loss + total_reg_loss

        sub_losses["1_total_epe"] = total_epe_loss
        sub_losses["2_reg"] = total_reg_loss

        return total_loss, sub_losses, pointwise_losses


class DispLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.weight_decay = 1e-4
        self.gt_interpolation = "bilinear"

        self.loss_weights = [1 / 16, 1 / 16, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1]
        self.loss_weights = [100 * weight for weight in self.loss_weights]
        self.reg_params = self.get_regularization_parameters(model)

    def get_regularization_parameters(self, model):
        reg_params = []
        for name, param in model.named_parameters():
            if (
                "pred" not in name
                and not name.endswith("bias")
                and not name.endswith("bn.weight")
                and param.requires_grad
            ):
                reg_params.append((name, param))

        print(
            "Applying regularization loss with weight decay {} on:".format(
                self.weight_decay
            )
        )
        for i, val in enumerate(reg_params):
            name, param = val
            print("\t#{} {}: {} ({})".format(i, name, param.shape, param.numel()))
        print()

        return reg_params

    def forward(self, sample, model_output):
        pointwise_losses = {}
        sub_losses = {}

        gt_disp = sample["gt_disp"]
        pred_disps_all = model_output["pred_disps_all"]

        total_epe_loss = 0
        total_reg_loss = 0

        for level, pred_disp in enumerate(pred_disps_all):
            with torch.no_grad():
                gt_disp_resampled = F.interpolate(
                    gt_disp,
                    size=pred_disp.shape[-2:],
                    mode=self.gt_interpolation,
                    align_corners=(
                        False if self.gt_interpolation != "nearest" else None
                    ),
                )

            epe_loss = epe(
                gt=gt_disp_resampled, pred=pred_disp, weight=self.loss_weights[level]
            )
            pointwise_epe_ = pointwise_epe(
                gt=gt_disp_resampled, pred=pred_disp, weight=self.loss_weights[level]
            )

            sub_losses["0_epe/level_%d" % level] = epe_loss
            pointwise_losses["0_epe/level_%d" % level] = pointwise_epe_

            total_epe_loss += epe_loss

        for name, param in self.reg_params:
            reg_loss = torch.sum(torch.mul(param, param)) / 2.0
            total_reg_loss += reg_loss

        total_reg_loss *= self.weight_decay

        total_loss = total_epe_loss + total_reg_loss

        sub_losses["1_total_epe"] = total_epe_loss
        sub_losses["2_reg"] = total_reg_loss

        return total_loss, sub_losses, pointwise_losses


class PhotometricLoss(nn.Module):
    def __init__(self, model, use_smoothness_loss=True, smoothness_weight=1.0):
        super().__init__()

        self.use_smoothness_loss = use_smoothness_loss
        self.smoothness_weight = smoothness_weight
        self.weight_decay = 1e-4
        self.gt_interpolation = "bilinear"

        self.loss_weights = [1 / 16, 1 / 16, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1]
        self.loss_weights = [100 * weight for weight in self.loss_weights]
        self.reg_params = self.get_regularization_parameters(model)

    def get_regularization_parameters(self, model):
        reg_params = []
        for name, param in model.named_parameters():
            if (
                "pred" not in name
                and not name.endswith("bias")
                and not name.endswith("bn.weight")
                and param.requires_grad
            ):
                reg_params.append((name, param))

        print(
            "Applying regularization loss with weight decay {} on:".format(
                self.weight_decay
            )
        )
        for i, val in enumerate(reg_params):
            name, param = val
            print("\t#{} {}: {} ({})".format(i, name, param.shape, param.numel()))
        print()

        return reg_params

    def forward(self, sample, model_output):
        pointwise_losses = {}
        sub_losses = {}

        images = sample["images_spatial"]
        pred_flows_all = model_output["pred_flows_all"]

        total_photo_loss = 0
        total_smoothness_loss = 0
        total_reg_loss = 0

        for level, pred_flow in enumerate(pred_flows_all):
            image_2_warped, warping_mask, pointwise_photo_loss, photo_loss = (
                photometric_loss(
                    image_1=images[0],
                    image_2=images[1],
                    pred_flow=pred_flow,
                    weight=self.loss_weights[level],
                )
            )
            sub_losses["0_photo/level_%d" % level] = photo_loss
            pointwise_losses["0_image2_warped/level_%d" % level] = image_2_warped
            pointwise_losses["1_warping_mask/level_%d" % level] = warping_mask
            pointwise_losses["2_photo/level_%d" % level] = pointwise_photo_loss
            total_photo_loss += photo_loss

            if self.use_smoothness_loss:
                pointwise_smoothness_loss, smoothness_loss_ = smoothness_loss(
                    pred_flow=pred_flow,
                    weight=self.loss_weights[level] * self.smoothness_weight,
                )
                sub_losses["1_smoothness/level_%d" % level] = smoothness_loss_
                pointwise_losses["3_smoothness/level_%d" % level] = (
                    pointwise_smoothness_loss
                )
                total_smoothness_loss += smoothness_loss_

        for name, param in self.reg_params:
            reg_loss = torch.sum(torch.mul(param, param)) / 2.0
            total_reg_loss += reg_loss

        total_reg_loss *= self.weight_decay

        total_loss = total_photo_loss + total_reg_loss + total_smoothness_loss

        sub_losses["2_total_photo"] = total_photo_loss
        sub_losses["3_total_smoothness"] = total_smoothness_loss
        sub_losses["2_reg"] = total_reg_loss

        return total_loss, sub_losses, pointwise_losses
