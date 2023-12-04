# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modified Eval_Hooks to add further logs
# Included: OnlineEvalHook, VideoEvalHook, ShiftEvalHook

import os.path as osp
import gc
import numpy as np
import os
from collections import OrderedDict
from functools import reduce
import cv2 as cv
from pathlib import Path

import torch
import torch.distributed as dist
import mmcv
from mmcv import tensor2imgs
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.utils import print_log
from prettytable import PrettyTable

ite_to_domain = {
    0: "clear",
    2976: "clear",
    5951: "clear",
    8926: "clear",
    11901: "25mm",
    14876: "25mm",
    17851: "25mm",
    20826: "50mm",
    23801: "50mm",
    26776: "50mm",
    29751: "75mm",
    32726: "75mm",
    35701: "75mm",
    38676: "100mm",
    41651: "100mm",
    44626: "100mm",
    47601: "200mm",
    50576: "200mm",
    53551: "200mm",
    56526: "100mm",
    59501: "100mm",
    62476: "100mm",
    65451: "75mm",
    68426: "75mm",
    71401: "75mm",
    74376: "50mm",
    77351: "50mm",
    80326: "50mm",
    83301: "25mm",
    86276: "25mm",
    89251: "25mm",
    92226: "clear",
    95201: "clear",
    98176: "clear",
}

domain_to_idx = {
    "clear": 0,
    "cloudy": 1,
    "overcast": 2,
    "small_rain": 3,
    "mid_rain": 4,
    "heavy_rain": 5,
}


class VideoEvalHook(_EvalHook):
    """Video evaluation consists in evaluating the same img that has been used for training."""

    greater_keys = ["mIoU", "mAcc", "aAcc"]

    def __init__(
        self,
        dataloaders,
        *args,
        by_epoch=False,
        efficient_test=False,
        work_dir=None,
        **kwargs,
    ):
        super().__init__(dataloaders[0], *args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self.work_dir = work_dir + "/imgs"
        self.before = False

    def save_prediction(self, img, img_meta, img_result, model):
        img = tensor2imgs(img, **img_meta["img_norm_cfg"])[0]

        dataset = self.dataloader.dataset
        h, w, _ = img_meta["img_shape"]
        img_show = img[:h, :w, :]

        ori_h, ori_w = img_meta["ori_shape"][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        img_pred = model.module.show_result(
            img_show,
            img_result,
            palette=dataset.PALETTE,
            show=False,
            out_file=None,
            opacity=1,
        )

        path = f"{self.work_dir}"
        Path(path).mkdir(parents=True, exist_ok=True)
        cv.imwrite(f'{path}/{img_meta["ori_filename"]}', img_pred)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""

        idx = runner.iter
        data = self.dataloader.dataset[idx]
        img = [data["img"][0][None, :, :, :]]
        img_metas = [[data["img_metas"][0].data]]
        gt = self.dataloader.dataset.get_idx_seg_map(idx)

        runner.model.eval()
        with torch.no_grad():
            results = runner.model(img, img_metas, return_loss=False)

        if self.dataloader.dataset.save_prediction:
            self.save_prediction(img[0], img_metas[0][0], results, runner.model)
        runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)
        # self.evaluate(runner, results, [gt])

    def evaluate(self, runner, results, gts=None):
        """Evaluate the results.

        Args:
            gts: groundtruth corresponding to the result
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """

        eval_res = self.evaluate_online(
            results,
            gts,
            self.dataloader.dataset,
            logger=runner.logger,
            work_dir=self.work_dir,
            t=runner.iter,
        )
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val

        runner.log_buffer.ready = True

    @staticmethod
    # Code obtained from: mmseg/datasets/custom.py
    def evaluate_online(
        results, gt_seg_maps, dataset, metric="mIoU", logger=None, work_dir=None, t=None
    ):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        assert len(results) == len(gt_seg_maps) and len(results) == 1

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ["mIoU", "mDice", "mFscore"]
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError("metric {} is not supported".format(metric))
        eval_results = {}
        if dataset.CLASSES is None:
            num_classes = len(reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(dataset.CLASSES)

        from mmseg.core import eval_metrics

        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            dataset.ignore_index,
            metric,
            label_map=dataset.label_map,
            reduce_zero_label=dataset.reduce_zero_label,
            work_dir=work_dir,
            t=t,
        )

        if dataset.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = dataset.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )

        # each class table
        ret_metrics.pop("aAcc", None)
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        ret_metrics_class.update({"Class": class_names})
        ret_metrics_class.move_to_end("Class", last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == "aAcc":
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column("m" + key, [val])

        print_log("per class results:", logger)
        print_log("\n" + class_table_data.get_string(), logger=logger)
        print_log("Summary:", logger)
        print_log("\n" + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == "aAcc":
                eval_results[key] = value / 100.0
            else:
                eval_results["m" + key] = value / 100.0

        ret_metrics_class.pop("Class", None)
        for key, value in ret_metrics_class.items():
            eval_results.update(
                {key + "." + str(name): value[idx] / 100.0 for idx, name in enumerate(class_names)}
            )

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results


class OnlineEvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support for Online training.

    Evaluate all the domains, regardless of the current domain, to measure forgetness.

    """

    greater_keys = ["mIoU", "mAcc", "aAcc"]

    def __init__(
        self,
        dataloaders,
        by_epoch=True,
        efficient_test=False,
        work_dir=None,
        num_imgs=50,
        **kwargs,
    ):
        super().__init__(dataloaders[0], by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self.dataloaders = dataloaders
        self.work_dir = work_dir + "/preds"
        self.first = True
        self.first_iter = True
        self.iter = False
        self.img_to_pred = {i for i in range(num_imgs)}

    def _should_evaluate(self, runner):
        if runner.iter == 0 and runner.epoch == 0 and self.first:
            return False
        elif not self.iter:
            return super()._should_evaluate(runner)

    def before_train_epoch(self, runner):
        if runner.iter == 0 and runner.epoch == 0 and self.first:
            self._do_evaluate(runner)
        super().before_train_epoch(runner)

    def after_run(self, runner):
        super().after_run(runner)

        try:
            if not runner.model.module.is_mad_training():
                return
        except:
            return

        runner.log_buffer.clear()
        probabilities, module_times = runner.model.module.get_mad_info()
        for i, (probs, times) in enumerate(zip(probabilities, module_times)):
            runner.log_buffer.output[f"MAD_distribution_end"] = probs
            runner.log_buffer.output[f"MAD_times_end"] = times
        runner.log_buffer.ready = True
        for hook in runner.hooks:
            if "Wandb" in hook.__class__.__name__:
                hook.after_train_epoch(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        self.first = False
        from mmseg.apis import single_gpu_test

        from run_experiments import DEBUG
        show = False
        out_dir = None
        if DEBUG:
            show = True
            out_dir = self.out_dir

        for dataloader in self.dataloaders:
            dataset_name = dataloader.dataset.name
            results = single_gpu_test(
                runner.model,
                dataloader,
                show=show,
                out_dir=out_dir,
                num_epoch=runner.iter,
                dataset_name=dataset_name,
                img_to_pred=self.img_to_pred,
                efficient_test=self.efficient_test,
            )
            self.evaluate(dataloader, runner, results, dataset_name)
            # ugly way to ensure ram does not crash having multiple val datasets
            gc.collect()

        try:
            if runner.model.module.is_mad_training():
                mad_dist, _ = runner.model.module.get_mad_info(softmax=False)
                np.save(f"{self.work_dir}/mad_hist_ite:{runner.iter}", mad_dist)
        except:
            pass

        if runner.iter == 0 and runner.epoch == 0:
            mode = runner.mode
            runner.mode = "val"
            for hook in runner.hooks:
                if "Logger" in hook.__class__.__name__:
                    hook.after_train_epoch(runner)
            runner.mode = mode
            # if runner.iter == 0 and runner.epoch == 0:
            runner.log_buffer.clear()
            runner.log_buffer.ready = False

    def evaluate(self, dataloader, runner, results, dataset_name=None):
        """Evaluate the results."""
        if (
            runner.model_name == "ModularEncoderDecoder"
            or runner.model_name == "Tent"
            or (
                runner.model_name == "DACS"
                and runner.model.module.model_type == "ModularEncoderDecoder"
            )
        ):
            main_results = runner.model.module.get_main_model()
            dataset_name = "unknown" if dataset_name is None else dataset_name

            eval_res = dataloader.dataset.evaluate(
                results, logger=runner.logger, **self.eval_kwargs
            )
            for name, val in eval_res.items():
                runner.log_buffer.output[f"{dataset_name}.decode_{main_results}.{name}"] = val
                runner.log_buffer.output[f"{dataset_name}.{name}"] = val
            runner.log_buffer.output[f"target_domain"] = runner.data_loader.dataset.domain

            runner.log_buffer.ready = True

        #!DEBUG
        elif (
                runner.model_name == "DACS"
                and runner.model.module.model_type == "OthersEncoderDecoder"
        ):
            dataset_name = "unknown" if dataset_name is None else dataset_name

            eval_res = dataloader.dataset.evaluate(
                results, logger=runner.logger, **self.eval_kwargs
            )
            for name, val in eval_res.items():
                runner.log_buffer.output[f"{dataset_name}.decode_4.{name}"] = val
                runner.log_buffer.output[f"{dataset_name}.{name}"] = val
            runner.log_buffer.output[f"target_domain"] = runner.data_loader.dataset.domain

            runner.log_buffer.ready = True

        else:
            super().evaluate(runner, results)


class ShiftEvalHook(OnlineEvalHook):
    def __init__(
        self,
        dataloaders,
        by_epoch=True,
        efficient_test=False,
        work_dir=None,
        num_imgs=10,
        domain_order=None,
        epoch_domain=None,
        **kwargs,
    ):
        super().__init__(
            dataloaders,
            by_epoch=by_epoch,
            efficient_test=efficient_test,
            work_dir=work_dir,
            num_imgs=num_imgs,
            **kwargs,
        )
        d = dict(
            clear=21216,
            cloudy=14127,
            overcast=8007,
            small_rain=7446,
            mid_rain=6375,
            heavy_rain=7191,
        )

        self.total_order = []
        for o in domain_order:
            aux = [o for _ in range(d[o] * epoch_domain)]
            self.total_order += aux

        self.ite_to_domain = dict()
        for i in range(0, len(self.total_order), self.interval):
            self.ite_to_domain[i] = self.total_order[i]

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        if runner.iter == 0:
            return

        self.first = False
        from mmseg.apis import single_gpu_test

        ite = domain_to_idx[self.ite_to_domain[runner.iter + 1]]
        dataloader = self.dataloaders[ite]
        dataset_name = dataloader.dataset.name
        results = single_gpu_test(
            runner.model,
            dataloader,
            show=True,
            out_dir=self.work_dir,
            num_epoch=runner.iter,
            dataset_name=dataset_name,
            img_to_pred=self.img_to_pred,
            efficient_test=self.efficient_test,
        )
        self.evaluate(dataloader, runner, results, dataset_name)
        # ugly way to ensure ram does not crash having multiple val datasets
        del results
        gc.collect()

        try:
            if runner.model.module.is_mad_training():
                mad_dist, _ = runner.model.module.get_mad_info(softmax=False)
                np.save(f"{self.work_dir}/mad_hist_ite:{runner.iter}", mad_dist)
        except:
            pass

        if runner.iter == 0 and runner.epoch == 0:
            mode = runner.mode
            runner.mode = "val"
            for hook in runner.hooks:
                if "Logger" in hook.__class__.__name__:
                    hook.after_train_epoch(runner)
            runner.mode = mode
            # if runner.iter == 0 and runner.epoch == 0:
            runner.log_buffer.clear()
            runner.log_buffer.ready = False


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ["mIoU", "mAcc", "aAcc"]

    def __init__(
        self,
        *args,
        by_epoch=False,
        efficient_test=False,
        work_dir=None,
        num_imgs=10,
        **kwargs,
    ):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self.work_dir = work_dir + "/preds"
        self.img_to_pred = set(np.random.choice(len(self.dataloader.dataset), num_imgs))

    def after_run(self, runner):
        super().after_run(runner)

        if not runner.model.module.is_mad_training():
            return

        runner.log_buffer.clear()
        probabilities, module_times = runner.model.module.get_mad_info()
        for i, (probs, times) in enumerate(zip(probabilities, module_times)):
            runner.log_buffer.output[f"MAD_distribution_end_{i}"] = probs
            runner.log_buffer.output[f"MAD_times_end_{i}"] = times
        runner.log_buffer.ready = True
        for hook in runner.hooks:
            if "Wandb" in hook.__class__.__name__:
                hook.after_train_epoch(runner)

    def _should_evaluate(self, runner):
        if runner.iter == 0:
            return True
        else:
            return super()._should_evaluate(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmseg.apis import single_gpu_test

        results = single_gpu_test(
            runner.model,
            self.dataloader,
            show=True,
            out_dir=self.work_dir,
            num_epoch=runner.iter,
            img_to_pred=self.img_to_pred,
            efficient_test=self.efficient_test,
        )
        runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)

        if runner.model.module.is_mad_training():
            mad_dist, _ = runner.model.module.get_mad_info(softmax=False)
            np.save(f"mad_hist_ite:{runner.iter}", f"{self.work_dir}/{mad_dist}")

        if runner.iter == 0:
            mode = runner.mode
            runner.mode = "train"
            for hook in runner.hooks:
                if "Logger" in hook.__class__.__name__:
                    hook.after_train_epoch(runner)
            runner.mode = mode
            runner.log_buffer.clear()
            runner.log_buffer.ready = False

    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        if runner.model_name == "ModularEncoderDecoder" or (
            runner.model_name == "DACS"
            and runner.model.module.model_type == "ModularEncoderDecoder"
        ):
            # Results here is a list of list where results[0] are the results for the first decoder
            main_results = runner.model.module.get_main_model()
            # extra_output = len(results) > runner.model.module.total_modules
            # main_results = len(results) if extra_output else main_results
            if isinstance(results[0], list):
                for i, decoder_results in enumerate(results):
                    eval_res = self.dataloader.dataset.evaluate(
                        decoder_results, logger=runner.logger, **self.eval_kwargs
                    )
                    for name, val in eval_res.items():
                        runner.log_buffer.output[f"decode_{i + 1}.{name}"] = val
                        if i == main_results - 1:
                            runner.log_buffer.output[name] = val
            else:
                eval_res = self.dataloader.dataset.evaluate(
                    results, logger=runner.logger, **self.eval_kwargs
                )
                for name, val in eval_res.items():
                    runner.log_buffer.output[f"decode_{main_results}.{name}"] = val
                    runner.log_buffer.output[name] = val

            runner.log_buffer.ready = True

            if self.save_best is not None:
                if self.key_indicator == "auto":
                    # infer from eval_results
                    self._init_rule(self.rule, list(eval_res.keys())[0])
                return eval_res[self.key_indicator]

            return None

        else:
            super().evaluate(runner, results)


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ["mIoU", "mAcc", "aAcc"]

    def __init__(self, *args, by_epoch=False, efficient_test=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, ".eval_hook")

        from mmseg.apis import multi_gpu_test

        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            efficient_test=self.efficient_test,
        )
        if runner.rank == 0:
            print("\n")
            runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
