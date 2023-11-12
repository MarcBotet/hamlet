from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from .modular_encoder_decoder import ModularEncoderDecoder


@SEGMENTORS.register_module()
class CustomEncoderDecoder(ModularEncoderDecoder):
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
        total_modules=0, #!DEBUG (231110)
        modules_update=None,
        batchnorm=False,
        alpha=None,
        mad_time_update=None,
        temperature=None,
        **cfg
        ):
        super(CustomEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            modular_training=modular_training,
            training_policy=training_policy,
            loss_weight=loss_weight,
            num_module=num_module,
            total_modules=total_modules,
            modules_update=modules_update,
            batchnorm=batchnorm,
            alpha=alpha,
            mad_time_update=mad_time_update,
            temperature=temperature,
        )
        a=1

