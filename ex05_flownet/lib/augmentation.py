import math

import numpy as np
from torchvision.transforms import ColorJitter
from PIL import Image
import cv2


class Bernoulli:
    def __init__(self, prob):
        self.prob = prob

    def sample(self, size=1):
        return np.random.binomial(n=1, p=self.prob, size=size)


class UniformBernoulli:
    def __init__(self, mean, spread, prob=1.0, exp=False):
        self.mean = mean
        self.spread = spread
        self.prob = prob
        if exp:
            self.sample = self.sample_exp
        else:
            self.sample = self.sample_noexp

    def sample_noexp(self, size=1):
        gate = Bernoulli(self.prob).sample(size)
        return gate * np.random.uniform(
            low=self.mean - self.spread, high=self.mean + self.spread, size=size
        )

    def sample_exp(self, size):
        gate = Bernoulli(self.prob).sample(size=1)
        return gate * np.exp(
            np.random.uniform(
                low=self.mean - self.spread, high=self.mean + self.spread, size=size
            )
        )


class FlowNetAugmentation:
    def __init__(self):
        self.out_size = (384, 768)

        # augment only the images, not the ground truth:
        self.augment_image_only = False

        # output normalizations:
        self.normalize_images = True
        self.normalize_mode = None

        # photometric augmentation params
        self.color_aug_prob = 1.0
        self.color_aug = ColorJitter(
            saturation=(0, 2), contrast=(0.01, 8), brightness=(0.01, 2.0), hue=0.5
        )
        self.asymmetric_color_aug_prob = 0.0
        self.eraser_aug_prob = 0.0

        # spatial augmentation params
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0
        self.max_stretch = 0.2

    def __call__(self, sample):
        images = sample["images"]  # 3, H, W

        gt_disp = (
            sample["gt_disp"]
            if ("gt_disp" in sample and not self.augment_image_only)
            else None
        )  # H, W
        gt_flow = (
            sample["gt_flow"]
            if ("gt_flow" in sample and not self.augment_image_only)
            else None
        )  # 2, H, W

        images = [np.transpose(image, [1, 2, 0]) for image in images]  # H, W, 3
        gt_flow = (
            np.transpose(gt_flow, [1, 2, 0]) if gt_flow is not None else None
        )  # H, W, 2

        images_spatial, gt_flow, gt_disp, spatial_params = self.spatial_transform(
            images, gt_flow, gt_disp
        )
        ht, wd, sht, swd, y0, x0 = spatial_params

        images = self.color_transform(images_spatial)
        images = self.eraser_transform(images)

        images = [np.transpose(image, [2, 0, 1]) for image in images]  # 3, H, W
        images = [np.ascontiguousarray(image) for image in images]
        images_spatial = [
            np.transpose(image_spatial, [2, 0, 1]) for image_spatial in images_spatial
        ]  # 3, H, W
        images_spatial = [
            np.ascontiguousarray(image_spatial) for image_spatial in images_spatial
        ]
        sample["images"] = images
        sample["images_spatial"] = images_spatial

        if gt_flow is not None:
            gt_flow = np.transpose(gt_flow, [2, 0, 1])  # 2, H, W
            gt_flow = np.ascontiguousarray(gt_flow)
            sample["gt_flow"] = gt_flow

        if gt_disp is not None:
            gt_disp = np.expand_dims(gt_disp, 0)
            sample["gt_disp"] = gt_disp

        sample["_orig_height"] = ht
        sample["_orig_width"] = wd
        sample["_spatial_aug_scaled_height"] = sht
        sample["_spatial_aug_scaled_width"] = swd
        sample["_spatial_aug_crop_y"] = y0
        sample["_spatial_aug_crop_x"] = x0

    def color_transform(self, images):
        if np.random.rand() < self.color_aug_prob:
            # asymmetric
            if np.random.rand() < self.asymmetric_color_aug_prob:
                images = [
                    np.array(self.color_aug(Image.fromarray(image)), dtype=np.uint8)
                    for image in images
                ]

            # symmetric
            else:
                num_images = len(images)
                image_stack = np.concatenate(images, axis=0)
                image_stack = np.array(
                    self.color_aug(Image.fromarray(image_stack)), dtype=np.uint8
                )
                images = np.split(image_stack, num_images, axis=0)

        if self.normalize_images:
            if self.normalize_mode == "imagenet":
                images = [(x / 255.0).astype(np.float32) for x in images]
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                images = [(image - mean) / std for image in images]
            else:
                images = [((x / 255.0) - 0.4).astype(np.float32) for x in images]

        return images

    def spatial_transform(self, images, flow, disp):
        if (
            images is not None
            and len(images) > 0
            and any([image is not None for image in images])
        ):
            ht, wd = [image for image in images if image is not None][0].shape[:2]
        elif flow is not None:
            ht, wd = flow.shape[:2]
        elif disp is not None:
            ht, wd = disp.shape[:2]

        if self.out_size is not None:
            cht, cwd = self.out_size
        else:
            cht = int(math.ceil(ht / 64.0) * 64.0)
            cwd = int(math.ceil(wd / 64.0) * 64.0)

        if np.random.rand() < self.spatial_aug_prob:
            min_scale = np.maximum((cht + 8) / float(ht), (cwd + 8) / float(wd))

            scale = (
                UniformBernoulli(mean=0.2, spread=0.4, exp=True).sample(1)[0]
                * UniformBernoulli(mean=0.0, spread=0.3, exp=True).sample(1)[0]
            )
            for i in range(5):
                if (
                    scale < 1.2 and np.random.rand() < 0.9
                ):  # mimic the validity check in the old tf code
                    scale = (
                        UniformBernoulli(mean=0.2, spread=0.4, exp=True).sample(1)[0]
                        * UniformBernoulli(mean=0.0, spread=0.3, exp=True).sample(1)[0]
                    )
                else:
                    break

            scale_x = scale
            scale_y = scale

            if np.random.rand() < self.stretch_prob:
                scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
                scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

            scale_x = np.clip(scale_x, min_scale, None)
            scale_y = np.clip(scale_y, min_scale, None)

            sht = None
            swd = None

            if (
                images is not None
                and len(images) > 0
                and any([image is not None for image in images])
            ):
                images = [
                    (
                        cv2.resize(
                            image,
                            None,
                            fx=scale_x,
                            fy=scale_y,
                            interpolation=cv2.INTER_LINEAR,
                        )
                        if image is not None
                        else None
                    )
                    for image in images
                ]
                sht, swd = [image for image in images if image is not None][0].shape[:2]

            if flow is not None:
                flow = cv2.resize(
                    flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST
                )
                flow = flow * [scale_x, scale_y]
                if sht is None or swd is None:
                    sht, swd = flow.shape[:2]

            if disp is not None:
                disp = cv2.resize(
                    disp, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST
                )
                disp = disp * scale_y
                if sht is None or swd is None:
                    sht, swd = disp.shape[:2]

            y0 = np.random.randint(0, sht - cht)
            x0 = np.random.randint(0, swd - cwd)

            if (
                images is not None
                and len(images) > 0
                and any([image is not None for image in images])
            ):
                images = [
                    image[y0 : y0 + cht, x0 : x0 + cwd] if image is not None else None
                    for image in images
                ]

            if flow is not None:
                flow = flow[y0 : y0 + cht, x0 : x0 + cwd]

            if disp is not None:
                disp = disp[y0 : y0 + cht, x0 : x0 + cwd]

        else:
            scale_x = cwd / wd
            scale_y = cht / ht

            if (
                images is not None
                and len(images) > 0
                and any([image is not None for image in images])
            ):
                images = [
                    (
                        cv2.resize(image, (cwd, cht), interpolation=cv2.INTER_LINEAR)
                        if image is not None
                        else None
                    )
                    for image in images
                ]

            if flow is not None:
                flow = cv2.resize(flow, (cwd, cht), interpolation=cv2.INTER_NEAREST)
                flow = flow * [scale_x, scale_y]

            if disp is not None:
                disp = cv2.resize(disp, (cwd, cht), interpolation=cv2.INTER_NEAREST)
                disp = disp * scale_y

            sht = cht
            swd = cwd
            x0 = 0
            y0 = 0

        return images, flow, disp, (ht, wd, sht, swd, y0, x0)

    def eraser_transform(self, images, bounds=(50, 100)):
        image_1 = images[0]
        image_2 = images[1]
        ht, wd = image_1.shape[:2]

        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(image_2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)

                min_x = max(0, x0 - dx // 2)
                max_x = min(wd - 1, x0 + dx // 2)

                min_y = max(0, y0 - dy // 2)
                max_y = min(ht - 1, y0 + dy // 2)

                image_2[min_y:max_y, min_x:max_x, :] = mean_color
            images[1] = image_2

        return images
