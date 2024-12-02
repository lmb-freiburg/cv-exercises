from lib.datasets.flyingthings3d import FlyingThings3D, DataConf


class FlyingThings3DTrain(FlyingThings3D):

    def __init__(self, root=None, aug_fcts=None, to_torch=False, verbose=True):
        split = "train"
        use_subset = True
        sample_confs = self._get_sample_confs()

        super().__init__(
            sample_confs,
            split,
            use_subset=use_subset,
            root=root,
            aug_fcts=aug_fcts,
            to_torch=to_torch,
            verbose=verbose,
        )

    def _get_sample_confs(self):
        sample_conf = {
            "images": [
                DataConf("frames_cleanpass", "left", 0),
                DataConf("frames_cleanpass", "right", 0),
            ],
            "gt_disp": DataConf("disparities", "left", 0),
        }

        return [sample_conf]


class FlyingThings3DTest(FlyingThings3D):

    def __init__(self, root=None, aug_fcts=None, to_torch=False, verbose=True):
        split = "test"
        use_subset = True
        sample_confs = self._get_sample_confs()

        super().__init__(
            sample_confs,
            split,
            use_subset=use_subset,
            root=root,
            aug_fcts=aug_fcts,
            to_torch=to_torch,
            verbose=verbose,
        )

    def _get_sample_confs(self):
        sample_conf = {
            "images": [
                DataConf("frames_cleanpass", "left", 0),
                DataConf("frames_cleanpass", "right", 0),
            ],
            "gt_disp": DataConf("disparities", "left", 0),
        }

        return [sample_conf]
