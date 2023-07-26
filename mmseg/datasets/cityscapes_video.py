from .builder import DATASETS
from .video import DatasetVideo


@DATASETS.register_module()
class CityscapesDatasetVideo(DatasetVideo):
    def __init__(self, **kwargs):
        super(CityscapesDatasetVideo, self).__init__(
            img_suffix=".png", seg_map_suffix=".png", **kwargs
        )
        self.color_seg_map = None
        self.idx = 0
        self.save_prediction = True
