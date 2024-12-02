import random

import numpy as np
import torch
import torch.nn as nn

from lib.vis import np3d


class UniformTorchCorr(nn.Module):
    def __init__(self, padding_mode="zeros"):
        super().__init__()
        self.padding_mode = padding_mode

    def forward(self, feat_ref, feat_src, offsets):
        """
        Correlate features feat_ref and feat_src for points in feat_src located at the specified offsets.

        Args:
            feat_ref: Reference feature map. NCHW.
            feat_src: Source feature map. NCHW.
            offsets: Offsets of the S correlation points in the source feature map. S2.

        Returns: Volume of correlation scores. NSHW.
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
        pad_top, pad_bot = max(0, -1 * torch.min(offsetsy).item()), max(
            0, torch.max(offsetsy).item()
        )
        pad_size = (pad_l, pad_r, pad_top, pad_bot)
        pad_fct = (
            nn.ConstantPad2d(pad_size, 0)
            if self.padding_mode == "zeros"
            else torch.nn.ReplicationPad2d(pad_size)
        )
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
        offsets_y, offsets_x = torch.meshgrid(
            offsets, offsets, indexing="ij"
        )  # both (S, S)
        self.offsets = torch.stack(
            (offsets_x.reshape(-1), offsets_y.reshape(-1)), 1
        )  # S**2, 2

    def __len__(self):
        return self.offsets.shape[0]


class DispSamplingPoints:
    @torch.no_grad()
    def __init__(self, steps=40, step_size=1):
        # START TODO #################
        # A DispNet uses 1D correlation along horizontal epipolar lines to the left
        # of the reference point.
        # Implement the sampling points accordingly. Store them in the attribute self.offsets.
        # The final shape of self.offsets should be (steps+1, 2).
        # Hint: The offsets in x direction can be computed similarly as in the FlowSamplingPoints.
        raise NotImplementedError
        # END TODO ###################

    def __len__(self):
        return self.offsets.shape[0]


class Corr:
    def __init__(self, steps=10, step_size=2, cuda_corr=False, corr_type="flow"):
        if corr_type == "flow":
            self.sampling_points = FlowSamplingPoints(steps=steps, step_size=step_size)
        else:
            self.sampling_points = DispSamplingPoints(steps=steps, step_size=step_size)

        if not cuda_corr:
            print("Using Python correlation layer.\n")
            self.corr_block = UniformTorchCorr(padding_mode="zeros")

        elif corr_type == "flow":
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

        else:
            raise NotImplementedError(
                "CUDA correlation layer for disparity correlation currently missing."
            )

        self.cuda_corr = cuda_corr
        self.num_sampling_points = (
            (2 * steps + 1) ** 2 if corr_type == "flow" else steps + 1
        )
        # TODO: return len(self.sampling_points) instead)

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

    @torch.no_grad()
    def visualize(
        self,
        image_ref,
        image_src,
        feat_ref,
        feat_src,
        num_points=1,
        x_refs=None,
        y_refs=None,
    ):
        corr = self(feat_ref=feat_ref, feat_src=feat_src)

        N, _, H, W = image_ref.shape
        _, _, h, w = feat_ref.shape

        if x_refs is None or y_refs is None:
            x_refs = [int(random.uniform(0.05, 0.95) * w) for _ in range(num_points)]
            y_refs = [int(random.uniform(0.05, 0.95) * h) for _ in range(num_points)]

        ref_vis = self._get_ref_vis(image_ref, N, W, w, x_refs, y_refs)  # N, 3, H, W
        src_vis = self._get_src_vis(
            image_src, corr, self.sampling_points, N, H, W, h, w, x_refs, y_refs
        )  # N, 3, H, W

        out = [ref_vis] + [src_vis]
        out = [torch.from_numpy(x).cuda() for x in out]
        return out

    @torch.no_grad()
    def _get_ref_vis(self, image_src, N, W, w, x_refs, y_refs):
        scale = float(W / w)

        x_refs_orig = [scale * (x_ref + 0.5) for x_ref in x_refs]
        y_refs_orig = [scale * (y_ref + 0.5) for y_ref in y_refs]

        image = image_src.detach().cpu().numpy()
        images_vis = []

        for n in range(N):
            image_n = image[n]  # 3HW

            markers_image = [
                {
                    "xy_pos": (x_refs_orig[idx], y_refs_orig[idx]),
                    "desc": "p_ref",
                    "desc_text_off": False,
                }
                for idx in range(len(x_refs_orig))
            ]
            out_format = {
                "type": "np",
                "mode": "RGB",
                "channels": "CHW",
                "dtype": "uint8",
            }

            image_n_vis = np3d(
                image_n,
                markers=markers_image,
                marker_radius=8,
                image_range_text_off=True,
                out_format=out_format,
            )
            images_vis.append(image_n_vis)

        ref_vis = np.stack(images_vis, 0)

        return ref_vis  # N, 3, H, W

    @torch.no_grad()
    def _get_src_vis(
        self, image_src, corr, sampling_points, N, _H, W, _h, w, x_refs, y_refs
    ):
        scale = float(W / w)

        image = image_src.detach().cpu().numpy()
        scores = corr.detach().cpu().numpy()

        images_vis = []

        for n in range(N):
            image_n = image[n]  # 3HW
            scores_n = [
                scores[n, :, y_refs[idx], x_refs[idx]] for idx in range(len(x_refs))
            ]

            markers_image = []

            for idx in range(len(x_refs)):
                x_ref = x_refs[idx]
                y_ref = y_refs[idx]

                score = scores_n[idx]
                steps = score.shape[0]
                x = sampling_points.offsets[:, 0] + x_ref
                y = sampling_points.offsets[:, 1] + y_ref

                score_is_constant = all([score[i] == score[0] for i in range(steps)])
                max_score_step = np.argmax(score)

                if not score_is_constant:  # TODO: what is this?
                    markers_image.append(
                        {
                            "xy_pos": (
                                scale * x[max_score_step],
                                scale * y[max_score_step],
                            ),
                            "marker_color": (0, 255, 0),
                            "score": np.nan,
                            "marker_radius": 2,
                        }
                    )

                max_score = None
                for step in range(0, steps):
                    score_cur = score[step]
                    max_score = (
                        score_cur
                        if ((max_score is None) or (score_cur > max_score))
                        else max_score
                    )

                for step in range(0, steps):
                    x_cur = x[step]
                    y_cur = y[step]
                    score_cur = score[step]

                    show_marker_text = score_cur != max_score

                    markers_image.append(
                        {
                            "xy_pos": (scale * x_cur, scale * y_cur),
                            "marker_color": (133, 153, 0),
                            "score": score_cur,
                            "desc_text_off": show_marker_text,
                            "score_text_off": show_marker_text,
                        }
                    )

            image_n_vis = np3d(
                image_n,
                markers=markers_image,
                marker_radius=4,
                marker_text_off=False,
                marker_cmap="plasma",
                image_range_text_off=True,
                marker_range_text_off=False,
                text_off=False,
                out_format={
                    "type": "np",
                    "mode": "RGB",
                    "channels": "CHW",
                    "dtype": "uint8",
                },
            )

            images_vis.append(image_n_vis)

        src_vis = np.stack(images_vis, 0)

        return src_vis  # N, 3, H, W
