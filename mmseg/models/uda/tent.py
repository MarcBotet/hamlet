# code from tent repository:
# https://github.com/DequanWang/tent

import time
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import load_checkpoint

from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import OtherDecorator


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@UDA.register_module()
class Tent(OtherDecorator):
    def __init__(self, **cfg):
        super(Tent, self).__init__(**cfg)

        self.model = build_segmentor(deepcopy(cfg["model"]))

        if "segmentator_pretrained" in cfg:
            checkpoint = load_checkpoint(
                self.model, cfg["student_pretrained"], map_location="cpu"
            )
            self.model.CLASSES = checkpoint["meta"]["CLASSES"]
            self.model.PALETTE = checkpoint["meta"]["PALETTE"]

        self.local_iter = 0

        self.prepare_tent()

    def prepare_tent(self):
        # https://github.com/DequanWang/tent

        # train mode, because tent optimizes the model to minimize entropy
        model = self.get_model()
        model.train()
        # disable grad, to (re-)enable only what tent updates
        model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics + layerNorm as cotta code
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    def get_param_list(self):
        model = self.get_model()
        param_list = []
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                for namep, p in m.named_parameters():
                    if namep in ["weight", "bias"]:  # weight is scale, bias is shift
                        param_list.append(p)

        return param_list

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        log_vars = self(data_batch, optimizer)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch["img_metas"]), time=elapsed
        )
        return outputs

    def forward_train(self, data, optimizer):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        model = self.get_model()

        # Train on source images
        clean_losses = self.get_model().forward_train(
            data["img"],
            data["img_metas"],
            data["gt_semantic_seg"],
            return_feat=True,
            confidence=True,
        )
        clean_losses.pop("features")
        buffer_loss, clean_log_vars = self._parse_losses(
            clean_losses, mode=[0, 0, 0, 1]
        )
        log_vars.update(clean_log_vars)

        logits = model.whole_inference(
            data["target_img"][0], data["target_img_metas"][0], True
        )

        tent_loss = softmax_entropy(logits[0]).mean()
        main_loss = tent_loss + buffer_loss
        main_loss.backward()

        log_vars.update(
            dict(
                tent_loss=tent_loss.item(),
                buffer_loss=buffer_loss.item(),
                main_loss=main_loss.item(),
            )
        )

        optimizer.step()
        optimizer.zero_grad()

        self.local_iter += 1

        return log_vars
