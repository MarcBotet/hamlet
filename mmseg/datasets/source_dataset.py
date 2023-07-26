from mmseg.datasets import DATASETS, OUDADataset
import numpy as np


@DATASETS.register_module()
class SourceDataset(OUDADataset):
    def __int__(self, source, target, cfg):
        super(SourceDataset, self).__init__(source, target, cfg)

    def get_dataset(self):
        return self.source

    def rcs_sampling(self, buffer_size):
        # loop to ensure not repeated images in the buffer
        indices = set()
        while len(indices) < buffer_size:
            c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
            f1 = np.random.choice(self.samples_with_class[c])
            indices.add(self.file_to_idx[f1])
        return list(indices)

    def __getitem__(self, idx):
        if self.rcs_enabled:
            s1 = self.get_rare_class_sample()
        else:
            s1 = self.source[idx]
        return s1

    def __len__(self):
        return len(self.source)
