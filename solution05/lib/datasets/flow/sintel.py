from lib.datasets.sintel import Sintel


class SintelFullTrain(Sintel):

    def __init__(
        self, renderpass="clean", root=None, aug_fcts=None, to_torch=False, verbose=True
    ):
        type_ = "flow"
        sample_confs = self._get_sample_confs()

        super().__init__(
            sample_confs=sample_confs,
            type_=type_,
            renderpass=renderpass,
            root=root,
            aug_fcts=aug_fcts,
            to_torch=to_torch,
            verbose=verbose,
        )

    def _get_sample_confs(self):
        sample_conf = {
            "images": [("images", i) for i in range(0, 2)],
            "gt_flow": ("flows", 0),
            "keyview_idx": 0,
        }

        return [sample_conf]


class SintelFullTrainFinal(SintelFullTrain):

    def __init__(self, root=None, aug_fcts=None, to_torch=False, verbose=True):
        super().__init__(
            renderpass="final",
            root=root,
            aug_fcts=aug_fcts,
            to_torch=to_torch,
            verbose=verbose,
        )


class SintelTrain(Sintel):

    def __init__(
        self, renderpass="clean", root=None, aug_fcts=None, to_torch=False, verbose=True
    ):
        type_ = "flow"
        sample_confs = self._get_sample_confs()
        sequences = [
            "temple_2",
            "bamboo_2",
            "market_6",
            "mountain_1",
            "ambush_5",
            "ambush_4",
            "bamboo_1",
            "market_2",
            "bandage_2",
            "ambush_2",
            "shaman_3",
            "ambush_7",
            "cave_4",
            "sleeping_1",
            "ambush_6",
            "shaman_2",
            "bandage_1",
            "cave_2",
            "temple_3",
            "market_5",
            "sleeping_2",
        ]

        super().__init__(
            sample_confs=sample_confs,
            type_=type_,
            renderpass=renderpass,
            sequences=sequences,
            root=root,
            aug_fcts=aug_fcts,
            to_torch=to_torch,
            verbose=verbose,
        )

    def _get_sample_confs(self):
        sample_conf = {
            "images": [("images", i) for i in range(0, 2)],
            "gt_flow": ("flows", 0),
            "keyview_idx": 0,
        }

        return [sample_conf]


class SintelTest(Sintel):

    def __init__(
        self, renderpass="clean", root=None, aug_fcts=None, to_torch=False, verbose=True
    ):
        type_ = "flow"
        sample_confs = self._get_sample_confs()
        sequences = ["alley_2", "alley_1"]

        super().__init__(
            sample_confs=sample_confs,
            type_=type_,
            renderpass=renderpass,
            sequences=sequences,
            root=root,
            aug_fcts=aug_fcts,
            to_torch=to_torch,
            verbose=verbose,
        )

    def _get_sample_confs(self):
        sample_conf = {
            "images": [("images", i) for i in range(0, 2)],
            "gt_flow": ("flows", 0),
            "keyview_idx": 0,
        }

        return [sample_conf]
