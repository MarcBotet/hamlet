# Obtained from: https://github.com/NVlabs/SegFormer
# Modifications from https://github.com/lhoyer/DAFormer: Model construction with loop
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# Modifications:
# - Add prediction after module 1
# - Add functions to freeze the selected models
# - Use IncrementalDecodeHead as parent class

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from .incremental_decode_head import IncrementalDecodeHead
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


@HEADS.register_module()
class SegFormerHead(IncrementalDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with
    Transformers
    """

    def __init__(self, **kwargs):
        super(SegFormerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        conv_kernel_size = decoder_params['conv_kernel_size']

        self.linear_c = {}
        for i, in_channels in zip(self.in_index, self.in_channels):
            self.linear_c[str(i)] = MLP(
                input_dim=in_channels, embed_dim=embedding_dim)
        self.linear_c = nn.ModuleDict(self.linear_c)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * len(self.in_index),
            out_channels=embedding_dim,
            kernel_size=conv_kernel_size,
            padding=0 if conv_kernel_size == 1 else conv_kernel_size // 2,
            norm_cfg=kwargs['norm_cfg'])

        self.linear_fuse_m1 = ConvModule(
            in_channels=embedding_dim * 1,
            out_channels=embedding_dim,
            kernel_size=conv_kernel_size,
            padding=0 if conv_kernel_size == 1 else conv_kernel_size // 2,
            norm_cfg=kwargs['norm_cfg'])

        self.linear_pred = nn.Conv2d(
            embedding_dim, self.num_classes, kernel_size=1)

        self.linear_pred_m1 = nn.Conv2d(
            embedding_dim, self.num_classes, kernel_size=1)

    def freeze_or_not_modules(self, modules: list, requires_grad=False, batchnorm=False):
        for m in modules:
            # Do not update this module if pre-training lightweight decoder
            requires_grad_modules = requires_grad if modules != [1] else False
            for p in self.linear_c[str(m-1)].parameters():
                p.requires_grad = requires_grad_modules
        for p in self.linear_fuse_m1.parameters():
            p.requires_grad = requires_grad
        for p in self.linear_fuse.parameters():
            p.requires_grad = requires_grad
        for p in self.linear_pred.parameters():
            p.requires_grad = requires_grad
        for p in self.linear_pred_m1.parameters():
            p.requires_grad = requires_grad

        if not batchnorm:
            for pm in self.linear_fuse.modules():
                if isinstance(pm, nn.BatchNorm2d):
                    if not requires_grad:
                        pm.eval()
                    else:
                        pm.train()
            for pm in self.linear_fuse_m1.modules():
                if isinstance(pm, nn.BatchNorm2d):
                    if not requires_grad:
                        pm.eval()
                    else:
                        pm.train()

    def forward(self, inputs, module=4):
        modules = list(range(module))

        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     print(f.shape)

        _c = {}
        for i in modules:
            # mmcv.print_log(f'{i}: {x[i].shape}, {self.linear_c[str(i)]}')
            _c[i] = self.linear_c[str(i)](x[i]).permute(0, 2, 1).contiguous()
            _c[i] = _c[i].reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if i != 0:
                _c[i] = resize(
                    _c[i],
                    size=x[0].size()[2:],
                    mode='bilinear',
                    align_corners=False)

        if module == 1:
            _c = self.linear_fuse_m1(torch.cat(list(_c.values()), dim=1))
        else:
            _c = self.linear_fuse(torch.cat(list(_c.values()), dim=1))

        if self.dropout is not None:
            x = self.dropout(_c)
        else:
            x = _c
        return self.linear_pred(x) if module == 4 else self.linear_pred_m1(x)
