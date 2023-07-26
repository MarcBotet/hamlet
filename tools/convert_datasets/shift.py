import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from PIL import Image
from pathlib import Path


def convert_to_train_id(file):
    # re-assign labels to match the format of Cityscapes
    pil_label = Image.open(file)
    label = np.asarray(pil_label)
    label = np.sum(label, axis=2)  # reshape from h x w x 3 -> h x w of 3 only the first one contain label the otehr 0
    shift_to_city = {
        1: 11,
        2: 13,
        4: 24,
        5: 17,
        6: 7,
        7: 7,
        8: 8,
        9: 21,
        10: 26,
        11: 12,
        12: 20,
        13: 23,
        18: 19,
        22: 22,
    }
    id_to_trainid = {
        7: 0,
        8: 1,
        11: 2,
        12: 3,
        13: 4,
        17: 5,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        31: 16,
        32: 17,
        33: 18
    }
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in shift_to_city.items():
        k_mask = label == k
        label_copy[k_mask] = id_to_trainid[v]
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    new_file = file.replace('.png', '_labelTrainIds.png')
    assert file != new_file
    sample_class_stats['file'] = new_file
    #new_file = new_file.replace('/data/datasets/', 'data/')
    #Path('/'.join(new_file.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    Image.fromarray(label_copy, mode='L').save(new_file)
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert SHIFT annotations to TrainIds')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=8, type=int, help='number of process')
    parser.add_argument('-vd', '--val_dir', help='validation folder path')
    parser.add_argument('-td', '--train_dir', help='training folder path')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    out_dir = args.out_dir
    mmcv.mkdir_or_exist(out_dir)

    val_dir = args.val_dir
    train_dir = args.train_dir

    poly_files = []

    def _scan(folder):
        for poly in mmcv.scandir(
                folder, suffix='front.png', recursive=True):
            poly_file = osp.join(folder, poly)
            poly_files.append(poly_file)

    _scan(val_dir)
    _scan(train_dir)

    poly_files = sorted(poly_files)

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                convert_to_train_id, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_to_train_id,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
