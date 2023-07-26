from abc import ABCMeta

import torch

from .decode_head import BaseDecodeHead


class IncrementalDecodeHead(BaseDecodeHead, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super(IncrementalDecodeHead, self).__init__(**kwargs)

    @staticmethod
    def entropy(probs):
        return torch.mean(-torch.sum(probs * torch.log(probs), 1))

    @staticmethod
    def confidence(probs):
        return torch.mean(torch.max(probs, 1)[0])

    def forward_train(
        self,
        inputs,
        img_metas,
        gt_semantic_seg,
        train_cfg,
        seg_weight=None,
        module=4,
        confidence=False,
    ):
        """Forward function for training."""
        seg_logits = self.forward(inputs, module)
        losses = [self.losses(seg_logits, gt_semantic_seg, seg_weight)]

        if confidence:
            probs = torch.nn.functional.softmax(seg_logits, dim=1)
            losses.append(self.entropy(probs))
            losses.append(self.confidence(probs))

        return losses

    def calculate_entropy(self, inputs, module):
        seg_logits = self.forward(inputs, module)
        probs = torch.nn.functional.softmax(seg_logits, dim=1)
        return self.entropy(probs).item(), self.confidence(probs).item()

    def forward_test(self, inputs, img_metas, test_cfg, module=4):
        """Forward function for testing.

        Include module to use

        """
        return self.forward(inputs, module)

    def to_freeze_or_not_modules(
        self, modules: list, requires_grad=False, batchnorm=False
    ):
        self.freeze_or_not_modules(modules, requires_grad, batchnorm)
