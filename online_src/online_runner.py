import time
import numpy as np

from mmcv.runner import EpochBasedRunner, get_host_info

from online_src.domain_indicator_orchestrator import DomainIndicator


class OnlineRunner(EpochBasedRunner):
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
        domain_indicator_args=None,
        cfg_lr=None,
        mode_train=True,
    ):
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
        self.domain_change = "static.decode_1.loss_seg"
        self.domain_indicator = DomainIndicator(**domain_indicator_args, **cfg_lr)
        self.initial_lr = cfg_lr["initial_lr"]
        self.policy_lr = cfg_lr["policy_lr"]
        self.max_lr = cfg_lr["max_lr"]
        self.lr_far_domain = cfg_lr["lr_far_domain"]
        self.mode_train = mode_train

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

    def get_lr_hook(self):
        for hook in self.hooks:
            if "Lr" in hook.__class__.__name__:
                return hook
        return None

    def get_lr_iters(self):
        return self.domain_indicator.iters

    def get_lr(self):
        if self.lr_far_domain:
            val = self.domain_indicator.get_domain_value()
            min_ratio, max_ratio = self.initial_lr, self.lr_far_domain
            return self.domain_indicator.linear_interpolation(val, [min_ratio, max_ratio], -1)
        else:
            return self.initial_lr

    def get_lr_schedule(self):
        if self.policy_lr == "constant":
            return "constant", self.get_lr()
        elif self.policy_lr == "adaptive_init":
            lr = self.domain_indicator.get_lr()
            cur_lr = self.optimizer.param_groups[0]["lr"]
            lr = min(lr + cur_lr, self.max_lr)
            lr_config = dict(
                policy="LinearDecay",
                min_lr=0.0,
                max_progress=self.domain_indicator.base_iters,
                by_epoch=False,
            )
        elif self.policy_lr == "adaptive_slope":
            lr = self.get_lr()
            lr_config = dict(
                policy="LinearDecay",
                min_lr=0.0,
                max_progress=self.domain_indicator.iters_train,
                by_epoch=False,
            )
        else:
            raise ValueError(f"policy lr {self.policy_lr} not valid")

        return lr_config, lr

    def replace_lr_hook(self):
        lr_config, lr = self.get_lr_schedule()

        # set lr
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        if lr_config == "constant":
            return
        idx = None
        for i, hook in enumerate(self.hooks):
            if "Lr" in hook.__class__.__name__:
                idx = i
                break
        if idx is not None:
            del self.hooks[idx]

            self.register_lr_hook(lr_config)

            for i, hook in enumerate(self.hooks):
                if "Lr" in hook.__class__.__name__:
                    hook.after_selected(self)

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.mode_train:
            kwargs["domain_indicator"] = self.domain_indicator.get_args()

            # train_step
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
        self.log_buffer.update(dict(time_speed=outputs["time"]))
        self.outputs = outputs
        if self.domain_indicator.domain_indicator:
            self.domain_indicator.add(outputs["log_vars"][self.domain_change])

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook("before_train_epoch")
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, target_data in enumerate(self.data_loader):
            # if i > 5: break #!DEBUG
            # if i > 100:
            #     break
            #     a=1
            source_data = self.next_source()
            data_batch = {
                **source_data,
                "target_img_metas": target_data["img_metas"],
                "target_img": target_data["img"],
            }
            self._inner_iter = i
            self.call_hook("before_train_iter")
            self.run_iter(data_batch, train_mode=True, **kwargs)

            changed, _ = self.domain_indicator.domain_changed()

            if changed:
                self.replace_lr_hook()

            self.log_buffer.update(
                dict(
                    domain_detected=self.domain_indicator.get_domain(),
                    is_training=self.domain_indicator.is_training() if self.mode_train else False,
                    domain_jump=self.domain_indicator.log_jump,
                    dynamic_dacs=self.domain_indicator.dacs,
                )
            )
            self.call_hook("after_train_iter")
            self._iter += 1

        self.call_hook("after_train_epoch")
        self._epoch += 1

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running."""
        assert isinstance(data_loaders, list)

        assert self._max_epochs is not None, "max_epochs must be specified during instantiation"

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == "train":
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info("Start running, host: %s, work_dir: %s", get_host_info(), work_dir)
        self.logger.info("workflow: %s, max: %d epochs", workflow, self._max_epochs)
        self.call_hook("before_run")
        wandb = self.get_wandb()

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(f'runner has no method named "{mode}" to run an ' "epoch")
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError("mode in workflow must be a str, but got {}".format(type(mode)))

                for _ in range(epochs):
                    if mode == "train" and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        if wandb:
            total_fps, std_fps = self.get_total_fps()
            wandb.run.summary["FPS"] = total_fps
            wandb.run.summary["FPS_std"] = std_fps
        # save fps array just in case in work_dirs
        times = np.array(self.time_elapsed)
        np.save(f"{self.work_dir}/fps_array.npy", times)
        self.call_hook("after_run")
