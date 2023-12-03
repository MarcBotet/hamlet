# -*- coding: utf-8 -*-

from typing import List

import torch.nn as nn


def freeze(module: nn.Module, *submodules: List[str]):
    a=1
    for param in module.parameters():
        param.requires_grad_(False)
        param.grad = None

    a=1
    # for n, param in module.parameters():
    #     a=1
    #     param.requires_grad_(False)
    #     param.grad = None
    for name, param in module.named_parameters():
        a=1
        if "adapter" not in name: continue
        param.requires_grad_(True)
        # param.grad = None


def unfreeze(module: nn.Module, *submodules: List[str]):
    if submodules:
        module = nn.ModuleList(
            [m for n, m in module.named_modules() if n in submodules]
        )
    for param in module.parameters():
        param.requires_grad_(True)

