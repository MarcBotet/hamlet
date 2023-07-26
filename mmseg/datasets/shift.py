from mmseg.datasets.cityscapes import CityscapesDataset

from .builder import DATASETS


@DATASETS.register_module()
class ShiftDataset(CityscapesDataset):
    """Shift dataset."""

    def __init__(self, **kwargs):
        super(ShiftDataset, self).__init__(
            img_suffix="_img_front.jpg",
            seg_map_suffix="_semseg_front_labelTrainIds.png",
            **kwargs
        )
        self.color_seg_map = None
