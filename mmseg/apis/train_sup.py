# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Add ddp_wrapper from mmgen
import os
import random
import warnings

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner

from mmseg.core import DistEvalHook, EvalHook, OnlineEvalHook
from mmseg.core.ddp_wrapper import DistributedDataParallelWrapper
from mmseg.core.evaluation.eval_hooks import ShiftEvalHook, VideoEvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger
from online_src.online_runner import OnlineRunner
from online_src.others_runner import OthersRunner


# def set_random_seed(seed=0, deterministic=True):
#     """Set random seed.

#     Args:
#         seed (int): Seed to be used.
#         deterministic (bool): Whether to set the deterministic option for
#             CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
#             to True and `torch.backends.cudnn.benchmark` to False.
#             Default: False.
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     if deterministic:
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False


def train_segmentor(
    model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None
):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    for param in model.model.backbone.parameters():
        param.requires_grad = False
    a=1

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True,
        )
        for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        use_ddp_wrapper = cfg.get("use_ddp_wrapper", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        if use_ddp_wrapper:
            mmcv.print_log("Use DDP Wrapper.", "mmseg")
            model = DistributedDataParallelWrapper(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters,
            )
    else:
        model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get("runner") is None:
        cfg.runner = {"type": "IterBasedRunner", "max_iters": cfg.total_iters}
        warnings.warn(
            "config is now expected to have a `runner` section, "
            "please set `runner` in your config.",
            UserWarning,
        )

    # if "others" in cfg:
    # if "tent" in cfg:
    #     other = "tent"
    #     optimizer = None
    # else:
    #     other = None
    #     optimizer = None

    #!DEBUG
    other = None
    args = cfg.runner.copy()
    source_dataloader = data_loaders.pop(0)
    default_args = dict(
        model=model,
        batch_processor=None,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta,
        source_dataloader=source_dataloader,
        samples_per_gpu=cfg.data.samples_per_gpu,
        other=other,
    )
    for name, value in default_args.items():
        args.setdefault(name, value)

    args.pop("type")
    runner = OthersRunner(**args)

    # register hooks
    runner.register_training_hooks(
        None, checkpoint_config=cfg.checkpoint_config, log_config=cfg.log_config
    )

    # elif "online" in cfg:
    #     # cityscapes numbers
    #     limit, threshold_src, threshold_max = (
    #         (75, 0.8, 2.55) if cfg.name_dataset != "shift2shift" else (200, 0.46, 1.85)
    #     )

    #     args = cfg.runner.copy()
    #     source_dataloader = data_loaders.pop(0)
    #     default_args = dict(
    #         model=model,
    #         batch_processor=None,
    #         optimizer=optimizer,
    #         work_dir=cfg.work_dir,
    #         logger=logger,
    #         meta=meta,
    #         source_dataloader=source_dataloader,
    #         samples_per_gpu=cfg.data.samples_per_gpu,
    #         domain_indicator_args=dict(
    #             domain_indicator=cfg.online.domain_indicator,
    #             limit=limit,
    #             threshold_src=threshold_src,
    #             threshold_max=threshold_max,
    #             far_domain=1.9,
    #             threshold=cfg.online.threshold_indicator,
    #             dynamic_dacs=cfg.uda.dynamic_dacs,
    #             base_iters=cfg.online.base_iters,
    #             max_iters=4450,
    #             reduce_training=cfg.online.reduce_training,
    #         ),
    #         cfg_lr=cfg["cfg_lr"],
    #         mode_train=cfg.online.mode_train,
    #     )
    #     for name, value in default_args.items():
    #         args.setdefault(name, value)

    #     args.pop("type")
    #     runner = OnlineRunner(**args)

    #     # register hooks
    #     runner.register_training_hooks(
    #         cfg.lr_config,
    #         cfg.optimizer_config,
    #         cfg.checkpoint_config,
    #         cfg.log_config,
    #         cfg.get("momentum_config", None),
    #     )
    # else:
    #     runner = build_runner(
    #         cfg.runner,
    #         default_args=dict(
    #             model=model,
    #             batch_processor=None,
    #             optimizer=optimizer,
    #             work_dir=cfg.work_dir,
    #             logger=logger,
    #             meta=meta,
    #         ),
    #     )

    #     # register hooks
    #     runner.register_training_hooks(
    #         cfg.lr_config,
    #         cfg.optimizer_config,
    #         cfg.checkpoint_config,
    #         cfg.log_config,
    #         cfg.get("momentum_config", None),
    #     )

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        samples = 1
        # if "online" in cfg:
        val_datasets = [build_dataset(val) for val in cfg.online.val]
        val_dataloader = [
            build_dataloader(
                ds,
                samples_per_gpu=samples,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False,
            )
            for ds in val_datasets
        ][0]
        #     eval_hook = OnlineEvalHook
        #     eval_hook = eval_hook if "video" not in cfg["mode"] else VideoEvalHook

        # else:
        # val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # val_dataloader = build_dataloader(
        #     val_dataset,
        #     samples_per_gpu=1,
        #     workers_per_gpu=cfg.data.workers_per_gpu,
        #     dist=distributed,
        #     shuffle=False,
        # )
        eval_hook = DistEvalHook if distributed else EvalHook
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = (
            cfg.runner["type"] != "IterBasedRunner" and "video" not in cfg["mode"]
        )
        eval_cfg["work_dir"] = cfg["work_dir"]

        if cfg['name_dataset'] == 'shift2shift' and cfg['mode'] == 'dacs_online':
            eval_cfg['epoch_domain'] = cfg['epoch_domain']
            eval_cfg['domain_order'] = cfg['domain_order']
            eval_cfg['by_epoch'] = False
            eval_hook = ShiftEvalHook

        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
