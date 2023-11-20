# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp
import tempfile
from pathlib import Path

import cv2 as cv

import mmcv
import numpy as np
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmseg.datasets import CityscapesDataset


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix=".npy", delete=False, dir=tmpdir
        ).name
    np.save(temp_file_name, array)
    return temp_file_name


def efficient_np2tmp(array, temp_file_name=None, tmpdir=None):
    """Saveusing efficient format"""

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix=".npz", delete=False, dir=tmpdir
        ).name
    np.savez_compressed(temp_file_name, array)
    return temp_file_name


def single_gpu_test(
    model,
    data_loader,
    show=False,
    out_dir=None,
    num_epoch=None,
    dataset_name=None,
    img_to_pred=None,
    efficient_test=False,
    opacity=1,
):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    if efficient_test:
        mmcv.mkdir_or_exist(".efficient_test")
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # deal with rainy dataset not using same resolution as CityScapes
            if dataset.gt_shape is not None:
                for metas in data["img_metas"]:
                    for ori_shape in metas.data[0]:
                        ori_shape["ori_shape"] = dataset.gt_shape
            result = model(return_loss=False, **data)

            if (
                (show or out_dir)
                and (img_to_pred is None or i in img_to_pred)
                and dataset.color_seg_map is not None
            ):
                img_tensor = data["img"][0]
                img_metas = data["img_metas"][0].data[0]
                color_gt_path = (
                    dataset.ann_dir
                    + "/"
                    + img_metas[0]["ori_filename"].replace(
                        dataset.img_suffix, dataset.color_seg_map
                    )
                )
                if isinstance(dataset, CityscapesDataset):
                    color_gt_path = color_gt_path.replace("data/", "/data/datasets/")

                if osp.exists(color_gt_path):
                    imgs = tensor2imgs(img_tensor, **img_metas[0]["img_norm_cfg"])
                    assert len(imgs) == len(img_metas)

                    for idx, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                        h, w, _ = img_meta["img_shape"]
                        img_show = img[:h, :w, :]

                        ori_h, ori_w = img_meta["ori_shape"][:-1]
                        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                        if out_dir:
                            if dataset_name is not None:
                                out_file = osp.join(
                                    out_dir,
                                    f"epoch_{num_epoch}",
                                    dataset_name,
                                    img_meta["ori_filename"],
                                )
                            else:
                                out_file = osp.join(
                                    out_dir,
                                    f"iter_{num_epoch}",
                                    img_meta["ori_filename"],
                                )

                        else:
                            out_file = None

                        Path(osp.dirname(out_file)).mkdir(parents=True, exist_ok=True)

                        _result = [result[idx]]
                        img_pred = model.module.show_result(
                            img_show,
                            _result,
                            # result,
                            palette=dataset.PALETTE,
                            show=False,
                            out_file=None,
                            opacity=opacity,
                        )
                        gt_image = cv.imread(color_gt_path)
                        cv.imwrite(out_file, img_pred)
                        source = out_file.replace(".png", "_source.png")
                        cv.imwrite(source, gt_image)
                        # a=1

                        # #!DEBUG
                        # import matplotlib.pyplot as plt
                        # from mmseg.models.utils.visualization import subplotimg

                        # ori_filename = color_gt_path.replace("gtFine", "leftImg8bit").replace("_color", "")
                        # ori_image = cv.imread(ori_filename)

                        # src_image = cv.imread(img_meta["filename"])
                        # if np.unique(src_image).shape == (2,):
                        #     src_image = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)

                        # fig, axs = plt.subplots(2, 2)
                        # subplotimg(axs[0][0], ori_image, "Original image")
                        # subplotimg(axs[1][0], src_image, "Source image")
                        # subplotimg(axs[0][1], gt_image, "Source mask")
                        # subplotimg(axs[1][1], img_pred, "Pred mask")

                        # plt.suptitle(img_meta["ori_filename"])

                        # out_debug_file = out_file.replace("epoch_", "epoch_debug_").replace("iter_", "iter_debug_")
                        # Path(osp.dirname(out_debug_file)).mkdir(parents=True, exist_ok=True)
                        # plt.savefig(out_debug_file)
                        # #------

        if isinstance(result, list):
            if efficient_test:
                result = [efficient_np2tmp(_, tmpdir='.efficient_test') for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = efficient_np2tmp(result, tmpdir='.efficient_test')
            results.append(result)
       
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    # for _ in range(len(results)):
    #     prog_bar.update()

    return results


def multi_gpu_test(
    model, data_loader, tmpdir=None, gpu_collect=False, efficient_test=False
):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    if efficient_test:
        mmcv.mkdir_or_exist(".efficient_test")
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir=".efficient_test") for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir=".efficient_test")
            results.append(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results
