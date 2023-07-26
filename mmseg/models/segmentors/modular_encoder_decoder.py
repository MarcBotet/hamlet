import numpy as np

from mmcv.runner import auto_fp16

from mmseg.core import add_prefix
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class ModularEncoderDecoder(EncoderDecoder):
    """Modular Encoder Decoder segmentors.

    Encoder and decoder are devided in m subparts. A module consist on one sub encoder and one
    sub decoder. A module M_i uses as an input M_i-1
    """

    def __init__(
        self,
        backbone,
        decode_head,
        neck=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        modular_training=None,
        training_policy=None,
        loss_weight=None,
        num_module=None,
        total_modules=None,
        modules_update=None,
        batchnorm=False,
        alpha=None,
        mad_time_update=None,
        temperature=None,
    ):
        super(ModularEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )

        self.modular_training = modular_training
        self.training_policy = training_policy
        self.loss_weight = loss_weight

        self.num_module = num_module
        self.main_model = num_module
        self.total_modules = total_modules
        if isinstance(modules_update, str):
            self.list_modules_to_update = np.load(modules_update)
        else:
            self.list_modules_to_update = modules_update
        self.iters = 0
        self.batchorm = batchnorm
        # MAD parameters
        self.mad_distribution = np.zeros(shape=total_modules)
        self.mad_module_times_selected = [0] * total_modules
        self.loss_t_2 = None
        self.loss_t_1 = None
        self.module_t_1 = None
        self.alpha = alpha

        self.mad_weights = [4.21, 4.83, 5.11, 5.73]
        self.mad_weights = self.softmax(np.array(self.mad_weights), temperature)
        self.mad_time_update = mad_time_update

    @staticmethod
    def softmax(x, t=1):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x / t) / np.sum(np.exp(x / t), axis=0)

    def extract_feat(self, img, module=4):
        """Extract features from images."""
        x = self.backbone(img, module)

        if self.with_neck:
            x = self.neck(x)
        return x

    def _decode_head_forward_train(
        self, x, img_metas, gt_semantic_seg, seg_weight=None, module=4, confidence=False
    ):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(
            x,
            img_metas,
            gt_semantic_seg,
            self.train_cfg,
            seg_weight,
            module,
            confidence=confidence,
        )
        if confidence:
            # remove the last element
            conf = loss_decode.pop()
            entr = loss_decode.pop()
            losses[f"decode_{module}.confidence"] = conf
            losses[f"decode_{module}.entropy"] = entr

        loss = loss_decode.pop()
        losses.update(add_prefix(loss, f"decode_{module}"))

        return losses

    def entropy_prediction(self, img, module):
        x = self.extract_feat(img, module)

        entr, conf = self.decode_head.calculate_entropy(x, module)

        return {f"confidence": conf, "entropy": entr}

    def freeze_or_not_modules(self, modules: list, requires_grad=False):
        self.backbone.freeze_or_not_modules(modules, requires_grad)
        self.decode_head.to_freeze_or_not_modules(modules, requires_grad, self.batchorm)

    def select_module_to_train(self, iter, last_module):
        """Select which modules will receive and update."""

        if self.modular_training:
            if self.training_policy == "RANDOM":
                # For reproducibility we pick the module from a pre-generated numpy array
                n = self.list_modules_to_update[iter]
            elif self.training_policy == 1:
                # Necessary to pre-train the lightweight decoder
                n = 1
            elif "MAD_UP" in self.training_policy:
                n = np.argmax(self.mad_distribution) + 1
                self.mad_module_times_selected[n - 1] += 1
            else:
                raise NotImplementedError("Unknown training policy {}".format(self.training_policy))

            loss_weight = [0 if i != n - 1 else 1 for i in range(len(self.loss_weight))]

            if self.training_policy == "MAD_UP" or self.training_policy == "RANDOM":
                not_freeze = list(range(n, self.num_module + 1))
                to_freeze = list(range(1, n))
            else:
                not_freeze = [n]
                to_freeze = [i + 1 for i, l in enumerate(loss_weight) if l == 0]

            self.freeze_or_not_modules(not_freeze, requires_grad=True)
            self.freeze_or_not_modules(to_freeze, requires_grad=False)

        else:
            n = [i for i, w in enumerate(self.loss_weight) if w != 0][-1] + 1

        return n, self.loss_weight.copy()

    def get_mad_info(self, softmax=True):
        if softmax:
            return self.softmax(self.mad_distribution), self.mad_module_times_selected
        else:
            return self.mad_distribution, self.mad_module_times_selected

    def get_main_model(self):
        return self.main_model

    def get_training_policy(self):
        return self.training_policy

    def is_mad_training(self):
        return "MAD" in str(self.training_policy)

    def logs_for_mad(self, log_vars):
        if self.is_mad_training():
            probabilities, module_times = self.get_mad_info()
            for i, (probs, times) in enumerate(zip(probabilities, module_times)):
                log_vars[f"MAD_distribution_{i + 1}"] = probs
                log_vars[f"MAD_times_{i + 1}"] = times
            for i, hist in enumerate(self.mad_distribution):
                log_vars[f"MAD_hist_{i + 1}"] = hist

        return log_vars

    def update_mad_histogram(self, loss_t_0, last_block_trained, iter):
        if not self.is_mad_training():
            return

        # handle first iterations
        if self.loss_t_1 is None or self.loss_t_2 is None or self.module_t_1 is None:
            self.loss_t_1 = loss_t_0
            self.loss_t_2 = loss_t_0
            self.module_t_1 = last_block_trained

        expected_loss = 2 * self.loss_t_1 - self.loss_t_2
        gain_loss = expected_loss - loss_t_0

        self.loss_t_2 = self.loss_t_1
        self.loss_t_1 = loss_t_0
        self.mad_distribution = (1 - self.alpha) * self.mad_distribution

        if self.mad_time_update:
            weight = self.mad_weights[self.module_t_1 - 1]
            weight = weight if weight > 0 else 1 - weight
            gain_loss *= weight

        self.mad_distribution[self.module_t_1 - 1] += self.alpha * gain_loss

        self.module_t_1 = last_block_trained

    def forward_train(
        self,
        img,
        img_metas,
        gt_semantic_seg,
        seg_weight=None,
        return_feat=False,
        module=4,
        confidence=False,
    ):
        """Forward function for training."""
        x = self.extract_feat(img, module)

        losses = dict()
        if return_feat:
            losses["features"] = x

        loss_decode = self._decode_head_forward_train(
            x, img_metas, gt_semantic_seg, seg_weight, module, confidence
        )
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg, seg_weight)
            losses.update(loss_aux)

        return losses

    @auto_fp16(apply_to=("img",))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, module=self.num_module, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        self.num_module, module_weight = self.select_module_to_train(self.iters, self.num_module)
        losses = self(**data_batch)

        loss, log_vars = self._parse_losses(losses, mode=module_weight)

        self.update_mad_histogram(loss, self.num_module, self.iters)

        log_vars = self.logs_for_mad(log_vars)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data_batch["img_metas"]))

        self.iters += 1

        return outputs
