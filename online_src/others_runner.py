import time
import numpy as np
import torch

from mmcv.runner import EpochBasedRunner, get_host_info

from online_src.domain_indicator_orchestrator import DomainIndicator


class OthersRunner(EpochBasedRunner):
    def __init__(
        self,
        model,
        batch_processor=None,
        optimizer=None,
        work_dir=None,
        logger=None,
        meta=None,
        max_iters=None,
        max_epochs=None,
        source_dataloader=None,
        samples_per_gpu=None,
        other=None,
    ):
        if other == "tent":
            param_list = model.module.get_param_list()
            optimizer = torch.optim.Adam(param_list, lr=1.0e-07, betas=(0.9, 0.999))
        super().__init__(
            model,
            batch_processor,
            optimizer,
            work_dir,
            logger,
            meta,
            max_iters,
            max_epochs,
        )
        self.source_dataloader = source_dataloader
        self.source_iterator = iter(self.source_dataloader)
        self.time_elapsed = []
        self.samples_per_gpu = samples_per_gpu
        self.other = other

    def next_source(self):
        try:
            source_sample = next(self.source_iterator)
        except StopIteration:
            self.source_iterator = iter(self.source_dataloader)
            source_sample = next(self.source_iterator)

        return source_sample

    def get_total_fps(self):
        time = np.array(self.time_elapsed)
        return (self._iter * self.samples_per_gpu) / np.sum(time), np.std(1 / time)

    def get_wandb(self):
        for hook in self.hooks:
            if "Wandb" in hook.__class__.__name__:
                return hook.wandb
        return None

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs
            )
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
            self.time_elapsed.append(outputs["time"])
        else:
            outputs = self.model.val_step(data_batch, **kwargs)
            self.time_elapsed.append(outputs["time"])
        if not isinstance(outputs, dict):
            raise TypeError(
                '"batch_processor()" or "model.train_step()"'
                'and "model.val_step()" must return a dict'
            )
        if "log_vars" in outputs:
            self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        if self.other == "tent":
            self.model.module.prepare_tent()
        self.mode = "train"
        self.data_loader = data_loader
        dataset = data_loader.dataset
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook("before_train_epoch")
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook("before_train_iter")

            if dataset.gt_shape is not None:
                for img_metas_data in data["img_metas"]:
                    for ori_shape in img_metas_data.data[0]:
                        ori_shape["ori_shape"] = dataset.gt_shape

            source_data = self.next_source()
            data_batch = {
                **source_data,
                "target_img_metas": data["img_metas"],
                "target_img": data["img"],
            }

            self.run_iter(data_batch, train_mode=True, **kwargs)

            self.call_hook("after_train_iter")
            self._iter += 1

        self.call_hook("after_train_epoch")
        self._epoch += 1

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)

        assert (
            self._max_epochs is not None
        ), "max_epochs must be specified during instantiation"

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == "train":
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info(
            "Start running, host: %s, work_dir: %s", get_host_info(), work_dir
        )
        self.logger.info("workflow: %s, max: %d epochs", workflow, self._max_epochs)
        self.call_hook("before_run")
        wandb = self.get_wandb()

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an ' "epoch"
                        )
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        "mode in workflow must be a str, but got {}".format(type(mode))
                    )

                for _ in range(epochs):
                    if mode == "train" and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)
                if wandb:
                    total_fps, std_fps = self.get_total_fps()
                    wandb.run.summary["FPS"] = total_fps
                    wandb.run.summary["FPS_std"] = std_fps

        time.sleep(1)  # wait for some hooks like loggers to finish
        if wandb:
            total_fps, std_fps = self.get_total_fps()
            wandb.run.summary["FPS"] = total_fps
            wandb.run.summary["FPS_std"] = std_fps
        # save fps array just in case in work_dirs
        times = np.array(self.time_elapsed)
        np.save(f"{self.work_dir}/fps_array.npy", times)
        self.call_hook("after_run")
