import torch
import torch.nn as nn

from lib.utils import get_pad_fct


class UniformTorchCorr(nn.Module):
    def __init__(self, padding_mode="zeros"):
        super().__init__()
        self.padding_mode = padding_mode

    def forward(self, feat_ref, feat_src, offsets):
        """
        Correlate features feat_ref and feat_src for points in feat_src
        located at the specified offsets.

        Args:
            feat_ref: Reference feature map. NCHW.
            feat_src: Source feature map. NCHW.
            offsets: Offsets of the S correlation points in the source feature map. S2.

        Returns:
            Volume of correlation scores. NSHW.
        """

        offsets_ = offsets.int()
        assert torch.all(offsets_ == offsets)

        offsetsx, offsetsy = offsets_[:, 0], offsets_[:, 1]
        N, _, H, W = feat_ref.shape
        device = feat_ref.device
        base_mask = torch.ones(
            size=(N, 1, H, W), dtype=torch.float32, device=device, requires_grad=False
        )

        pad_l, pad_r = max(0, -1 * torch.min(offsetsx).item()), max(
            0, torch.max(offsetsx).item()
        )
        pad_top, pad_bot = (
            max(0, -1 * torch.min(offsetsy).item()),
            max(0, torch.max(offsetsy).item()),
        )
        pad_size = (pad_l, pad_r, pad_top, pad_bot)
        pad_fct = get_pad_fct(self.padding_mode, pad_size)
        feat_src = pad_fct(feat_src)
        base_mask = pad_fct(base_mask)

        outputs = []
        masks = []
        for dx, dy in zip(offsetsx, offsetsy):
            prod = (
                feat_ref
                * feat_src[
                    :, :, pad_top + dy : pad_top + dy + H, pad_l + dx : pad_l + dx + W
                ]
            )  # N, C, H, W
            corr = torch.sum(prod, 1, keepdim=True)  # N, 1, H, W
            mask = base_mask[
                :, :, pad_top + dy : pad_top + dy + H, pad_l + dx : pad_l + dx + W
            ]
            outputs.append(corr)
            masks.append(mask)
        corr = torch.cat(outputs, 1)  # N, S, H, W
        mask = torch.cat(masks, 1)  # N, S, H, W

        return corr, mask


class FlowSamplingPoints:
    @torch.no_grad()
    def __init__(self, steps=10, step_size=2):
        offsets = (
            torch.linspace(-steps, steps, 2 * steps + 1, dtype=torch.int32) * step_size
        )  # S = 2*steps+1
        offsets_y, offsets_x = torch.meshgrid(offsets, offsets)  # both (S, S)
        self.offsets = torch.stack(
            (offsets_x.reshape(-1), offsets_y.reshape(-1)), 1
        )  # S**2, 2

    def __len__(self):
        return self.offsets.shape[0]


class Corr:
    def __init__(self, steps=10, step_size=2, cuda_corr=False):
        if not cuda_corr:
            print("Using Python correlation layer.\n")
            self.sampling_points = FlowSamplingPoints(steps=steps, step_size=step_size)
            self.corr_block = UniformTorchCorr(padding_mode="zeros")
        else:
            print("Using CUDA correlation layer.\n")
            # noinspection PyUnresolvedReferences
            from spatial_correlation_sampler import SpatialCorrelationSampler

            self.corr_block = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=21,
                stride=1,
                padding=0,
                dilation=1,
                dilation_patch=2,
            )

        self.cuda_corr = cuda_corr
        self.num_sampling_points = (2 * steps + 1) ** 2

    def __call__(self, feat_ref, feat_src):
        if not self.cuda_corr:
            offsets = self.sampling_points.offsets.to(feat_ref.device)
            corr, _ = self.corr_block(
                feat_ref=feat_ref, feat_src=feat_src, offsets=offsets
            )
        else:
            corr = self.corr_block(feat_ref, feat_src)
            N, c1, c2, h, w = corr.shape
            corr = corr.reshape(N, c1 * c2, h, w)

        return corr

    def __len__(self):
        return self.num_sampling_points
