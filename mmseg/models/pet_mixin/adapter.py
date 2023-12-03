from typing import Dict, List

import torch
import torch.nn as nn

from mmseg.models.builder import BACKBONES
from mmseg.models.pet import Adapter, Conv2dAdapter


NAME_SEP = "/"

def normalize_name(name):
    return name.replace(".", NAME_SEP)

def denormlize_name(name):
    return name.replace(NAME_SEP, ".")

def get_submodule(
    module: nn.Module,
    name: str,
    default: nn.Module = nn.Identity(),
):
    names = name.split(NAME_SEP)
    while names:
        module = getattr(module, names.pop(0), default)
    return module


class AdapterMixin:
    adapters: nn.ModuleDict


    def attach_adapter(self, **kwargs: Dict[str, nn.Module]):
        a=1
        if not isinstance(getattr(self, "adapters", None), nn.ModuleDict):
            self.adapters = nn.ModuleDict()
        
        for name, adapter in kwargs.items():
            name = normalize_name(name)
            self.adapters.add_module(name, adapter)


    def detach_adapter(self, *names: List[str]):
        adapters = {}
        if not hasattr(self, "adapters"):
            return adapters

        names = names if names else map(denormlize_name, self.adapters.keys())
        for name in names:
            adapters[name] = self.adapters.pop(normalize_name(name))
        return adapters

    def adapt_module(self, name: str, input: torch.Tensor, **kwargs):
        name = normalize_name(name)
        module = get_submodule(self, name)

        # if not isinstance(module, (Adapter, Conv2dAdapter)): #근데이건 어떻게하는거지? module이 adapter 여야한다는 건가? 근데 나는 MLP에 adapter를 붙이고 싶은건데
        #     assert kwargs == {}, f"Unknown kwargs: {kwargs.keys()}" #일단은 kwargs를 {}로 설정해서 이거 넘기고
        #     # 아예 이 라인을 빼면 되지 않을까? ㅋㅋ 왜 있는거지

        if hasattr(self, "adapters") and name in self.adapters: #이건 어떻게하는지 알겠음
            return self.adapters[name](module, input, **kwargs)
        return module(input, **kwargs)

