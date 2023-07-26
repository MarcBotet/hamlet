from mmcv.runner import LrUpdaterHook, HOOKS
from typing import Callable, List, Optional, Union

from mmcv.runner.hooks.lr_updater import annealing_linear


@HOOKS.register_module()
class LinearDecayLrUpdaterHook(LrUpdaterHook):
    """Linear annealing LR Scheduler decays the learning rate of each parameter
    group linearly.
    Args:
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(
        self,
        min_lr: Optional[float] = None,
        min_lr_ratio: Optional[float] = None,
        max_progress: Optional[int] = None,
        **kwargs
    ):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.max_progress = max_progress
        super().__init__(**kwargs)

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float):
        progress = runner.get_lr_iters()
        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr  # type:ignore
        new_lr = annealing_linear(base_lr, target_lr, progress / self.max_progress)
        return new_lr if new_lr > 0 else 0

    def after_selected(self, runner):
        for group in runner.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lr = [group["lr"] for group in runner.optimizer.param_groups]
