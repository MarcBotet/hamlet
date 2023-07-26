from .builder import DATASETS
from .video import DatasetVideo
from ..utils import get_root_logger


@DATASETS.register_module()
class ShiftDatasetVideo(DatasetVideo):
    def __init__(self, **kwargs):
        super(ShiftDatasetVideo, self).__init__(
            img_suffix="_img_front.jpg",
            seg_map_suffix="_semseg_front_labelTrainIds.png",
            **kwargs
        )
        self.color_seg_map = None
        self.idx = 0
