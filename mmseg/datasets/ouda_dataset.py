# get_rare_class_sample is modified to from : https://github.com/lhoyer/DAFormer
from mmseg.datasets import DATASETS, UDADataset
import numpy as np
import torch


@DATASETS.register_module()
class OUDADataset(UDADataset):
    def __int__(self, source, target, cfg):
        super(OUDADataset, self).__init__(source, target, cfg)

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = self.source[i1]
        if self.rcs_min_crop_ratio > 0:
            for _ in range(10):
                n_class = torch.sum(s1["gt_semantic_seg"].data == c)
                # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                # Sample a new random crop from source image i1.
                # Please note, that self.source.__getitem__(idx) applies the
                # preprocessing pipeline to the loaded image, which includes
                # RandomCrop, and results in a new crop of the image.
                s1 = self.source[i1]

        return s1

    def __getitem__(self, idx):
        if self.rcs_enabled:
            s1 = self.get_rare_class_sample()
            s2 = self.target[idx]

        else:
            s1 = self.source[idx % len(self.source)]
            s2 = self.target[idx]

        return {**s1, "target_img_metas": s2["img_metas"], "target_img": s2["img"]}

    def __len__(self):
        return len(self.target)
