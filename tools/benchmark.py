# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add FPS during UDA training

import time

import numpy as np
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_optimizer

from mmseg.datasets import build_dataloader, build_dataset

import torch

import argparse
import json
import logging

from experiments import generate_experiment_cfgs
from mmcv import Config, get_logger

from mmseg.models import build_segmentor
from mmseg.models.builder import build_train_model


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    args = parser.parse_args()
    return args


def perform_benchmark(cfg):
    get_logger('mmseg', log_level=logging.ERROR)

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)

    samples_per_gpu = args.num_samples

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    # load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    total_iters = args.total_iters

    # benchmark with 200 image and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = ((i + 1 - num_warmup) * samples_per_gpu) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {total_iters}], '
                      f'fps: {fps:.2f} img / s')

        if (i + 1) == total_iters:
            fps = ((i + 1 - num_warmup) * samples_per_gpu) / pure_inf_time
            print(f'Overall fps inference: {fps:.2f} img / s')
            break
    #print(f'whyyyy {len(data_loader)}')


def perform_benchmark_training(cfg):
    assert 'uda' in cfg

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False

    cfg.uda.benchmark = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.train)

    samples_per_gpu = args.num_samples

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        drop_last=True)

    # build the model
    model = build_train_model(
        cfg, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    model = MMDataParallel(model, device_ids=[0])

    model.train()

    # the first several iterations may be very slow so skip them
    num_warmup = 10
    pure_inf_time = 0
    total_iters = args.total_iters

    times = []

    # benchmark with 200 image and take the average
    for i, data in enumerate(data_loader):

        elapsed = model.train_step(data, optimizer)['time']

        if i >= num_warmup:
            times.append(elapsed)

        if (i + 1) == total_iters:
            times = np.array(times)
            fps = ((len(times)) * samples_per_gpu) / np.sum(times)  # not mean cause we may use batch size > 1
            print(f'Overall fps UDA: {fps:.2f} img / s')
            print(f'STD of the time of each training step: {np.std(times)}')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp',
        nargs='?',
        type=int,
        default=-1,
        help='Experiment id as defined in experiment.py',
    )
    parser.add_argument(
        '--log-interval', type=int, default=500, help='interval of logging')

    parser.add_argument(
        '--total_iters', type=int, default=50, help='interval of logging')

    parser.add_argument(
        '--num-samples', type=int, default=1, help='interval of logging')

    args = parser.parse_args()

    get_logger('mmseg', log_level=logging.ERROR)

    cfgs = generate_experiment_cfgs(args.exp)
    for cfg in cfgs:
        with open('configs/tmp_param.json', 'w') as f:
            json.dump(cfg, f)
        cfg = Config.fromfile('configs/tmp_param.json')

        split_line = '=' * 30
        print(f'({cfg["name_encoder"]}, {cfg["name_decoder"]})')
        print(split_line)
        #perform_benchmark(cfg)
        if 'uda' in cfg:
            perform_benchmark_training(cfg)
        print(split_line)
