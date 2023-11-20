from copy import deepcopy

from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint

from mmseg.models import BaseSegmentor, build_segmentor


def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module


class UDADecorator(BaseSegmentor):
    def __init__(self, **cfg):
        super(BaseSegmentor, self).__init__()

        self.model = build_segmentor(deepcopy(cfg["model"]))
        self.train_cfg = cfg["model"]["train_cfg"]
        self.test_cfg = cfg["model"]["test_cfg"]
        self.num_classes = cfg["model"]["decode_head"]["num_classes"]

    def get_model(self):
        return get_module(self.model)

    def extract_feat(self, img):
        """Extract features from images."""
        return self.get_model().extract_feat(img)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_model().encode_decode(img, img_metas)

    def forward_train(
        self,
        img,
        img_metas,
        gt_semantic_seg,
        target_img,
        target_img_metas,
        return_feat=False,
    ):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=return_feat
        )
        return losses

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        return self.get_model().inference(img, img_meta, rescale)

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        return self.get_model().simple_test(img, img_meta, rescale)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        return self.get_model().aug_test(imgs, img_metas, rescale)


class ModularUDADecorator(UDADecorator):
    """Decorator for UDA using Modular models."""

    def __init__(self, **cfg):
        super(ModularUDADecorator, self).__init__(**cfg)

        self.model_type = cfg["model"]["type"]
        self.num_modules = 0
        if "num_modules" in cfg["model"]:
            self.num_modules = cfg["model"]["num_modules"]
        if "segmentator_pretrained" in cfg:
            checkpoint = load_checkpoint(
                self.model, cfg["student_pretrained"], map_location="cpu"
            )
            self.model.CLASSES = checkpoint["meta"]["CLASSES"]
            self.model.PALETTE = checkpoint["meta"]["PALETTE"]

    def freeze_or_not_modules(self, modules: list, requires_grad=False):
        self.get_model().freeze_or_not_modules(modules, requires_grad)

    def get_mad_info(self, softmax=True):
        return self.get_model().get_mad_info(softmax)

    def get_main_model(self):
        return self.get_model().get_main_model()

    def get_training_policy(self):
        return self.get_model().get_training_policy()

    def is_mad_training(self):
        return self.get_model().is_mad_training()


class OtherDecorator(UDADecorator):
    """Decorator for UDA for the other models."""

    def __init__(self, **cfg):
        super(OtherDecorator, self).__init__(cfg)

    def get_main_model(self):
        return self.get_model().get_main_model()


class CustomUDADecorator(UDADecorator):
    def __init__(self, **cfg):
        super(CustomUDADecorator, self).__init__(**cfg)

        self.model_type = cfg["model"]["type"]
        if "segmentator_pretrained" in cfg:
            checkpoint = load_checkpoint(
                self.model, cfg["student_pretrained"], map_location="cpu"
            )
            self.model.CLASSES = checkpoint["meta"]["CLASSES"]
            self.model.PALETTE = checkpoint["meta"]["PALETTE"]


    def get_main_model(self):
        return self.get_model() #.get_main_model()

