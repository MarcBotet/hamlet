# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Provide args as argument to main()
# - Snapshot source code
# - Build UDA model instead of regular one

import argparse
import copy
import os
import os.path as osp
import sys
import time

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from mmcv.runner import load_checkpoint

from mmseg import __version__

#!DEBUG
from mmseg.apis import set_random_seed, train_segmentor
# ----- below is for supervised encoder training (train_segmentor_sup)
# from run_experiments import CUSTOM
# if CUSTOM:
#     from mmseg.apis import set_random_seed
#     from mmseg.apis import set_random_seed, train_segmentor
#     # from mmseg.apis import train_segmentor_sup as train_segmentor
# else:
#     from mmseg.apis import set_random_seed, train_segmentor

from mmseg.datasets import build_dataset
from mmseg.models.builder import build_train_model
from mmseg.utils import collect_env, get_root_logger
from mmseg.utils.collect_env import gen_code_archive

from online_src.buffer import create_buffer


def parse_args(args):
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main(args):

    args = parse_args(args)

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    a=1
    from run_experiments import CUSTOM
    if CUSTOM:
        cfg["custom"] = cfg["uda"].copy()

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    cfg.model.train_cfg.work_dir = cfg.work_dir
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # snapshot source code
    # gen_code_archive(cfg.work_dir)
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is None and 'seed' in cfg:
        args.seed = cfg['seed']
    if 'deterministic' in cfg:
        args.deterministic = cfg['deterministic']
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.splitext(osp.basename(args.config))[0]

    # ----- Build Training Model (default)
    model = build_train_model(
        cfg, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    if ('uda' not in cfg or not 'segmentator_pretrained' in cfg['uda']) and cfg['segmentator_pretrained'] is None:
        model.init_weights()
    elif cfg['segmentator_pretrained'] is not None and 'uda' not in cfg:
        checkpoint = load_checkpoint(model, cfg['segmentator_pretrained'], map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
        logger.info('Init weights using a warmup model')
    else:
        logger.info('Init weights using a warmup model')

    logger.info(model)

    if 'online_old' in cfg:
        # ugly walkaround because json does not store tuples
        for train in cfg.online.train:
            for s_pipe in train.source.pipeline:
                for key, val in s_pipe.items():
                    if isinstance(val, list) and len(val) == 2:
                        s_pipe[key] = tuple(val)
            for s_pipe in train.target.pipeline:
                for key, val in s_pipe.items():
                    if isinstance(val, list) and len(val) == 2:
                        s_pipe[key] = tuple(val)

        for val in cfg.online.val:
            for s_pipe in val.pipeline:
                for key, val in s_pipe.items():
                    if isinstance(val, list) and len(val) == 2:
                        s_pipe[key] = tuple(val)

        cfg.workflow = [tuple(a) for a in cfg.workflow]

        datasets = [build_dataset(train) for train in cfg.online.train]
        classes = datasets[0].CLASSES
        palette = datasets[0].PALETTE
    elif 'online' in cfg:
        # ugly walkaround because json does not store tuples
        for train in cfg.online.train:
            if train.type == 'SourceDataset':
                train = train.source
            for s_pipe in train.pipeline:
                for key, val in s_pipe.items():
                    if isinstance(val, list) and len(val) == 2:
                        s_pipe[key] = tuple(val)

        for val in cfg.online.val:
            for s_pipe in val.pipeline:
                for key, val in s_pipe.items():
                    if isinstance(val, list) and len(val) == 2:
                        s_pipe[key] = tuple(val)

        cfg.workflow = [tuple(a) for a in cfg.workflow]

        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        if 'others' in cfg:
            test_pipeline = [
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(1024, 512),
                    #img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(type='RandomFlip'),
                        dict(type='Normalize', **img_norm_cfg),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img']),
                    ])
            ]

            for train in cfg.online.train:
                if train.type == 'SourceDataset':
                    continue
                train.pipeline = test_pipeline

        datasets = [build_dataset(train) for train in cfg.online.train]

        classes = datasets[0].CLASSES
        palette = datasets[0].PALETTE

        if cfg.online.buffer is not None:
            datasets[0] = create_buffer(datasets[0], cfg.online.buffer_policy, cfg.online.buffer)
    else:
        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))
        classes = datasets[0].CLASSES
        palette = datasets[0].PALETTE
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=classes,
            PALETTE=palette)
    # add an attribute for visualization convenience
    model.CLASSES = classes
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main(sys.argv[1:])
