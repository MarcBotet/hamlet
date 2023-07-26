import os.path as osp

import mmcv
from mmseg.datasets.cityscapes import CityscapesDataset
import numpy as np
from mmcv.utils import print_log

from .builder import DATASETS
from ..utils import get_root_logger


@DATASETS.register_module()
class DatasetVideo(CityscapesDataset):
    """Video dataset.

    Read the dataset by order

    """

    def __init__(self, img_suffix, seg_map_suffix, **kwargs):
        super(DatasetVideo, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs
        )
        self.color_seg_map = "_gtFine_color.png"
        self.idx = 0
        self.save_prediction = False

    @staticmethod
    def _convert_to_label_id(result):
        """Convert trainId to id for cityscapes."""
        if isinstance(result, str):
            result = np.load(result)
        import cityscapesscripts.helpers.labels as CSLabels

        result_copy = result.copy()
        for trainId, label in CSLabels.trainId2label.items():
            result_copy[result == trainId] = label.id

        return result_copy

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info["ann"] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info["ann"] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(
            f"Loaded {len(img_infos)} images from {img_dir}", logger=get_root_logger()
        )
        return sorted(img_infos, key=lambda x: int(x["filename"].split(".")[0]))

    def get_idx_seg_map(self, idx, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        img_info = self.img_infos[idx]
        seg_map = osp.join(self.ann_dir, img_info["ann"]["seg_map"])
        if efficient_test:
            gt_seg_map = seg_map
        else:
            gt_seg_map = mmcv.imread(seg_map, flag="unchanged", backend="pillow")
        return gt_seg_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        self.idx = idx
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)
