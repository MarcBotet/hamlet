from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
# from .modular_encoder_decoder import ModularEncoderDecoder


@SEGMENTORS.register_module()
class OthersEncoderDecoder(EncoderDecoder):
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
        # modular_training=None,
        # training_policy=None,
        # loss_weight=None,
        # num_module=None,
        # modules_update=None,
        # batchnorm=False,
        # alpha=None,
        # mad_time_update=None,
        # temperature=None,
        **cfg
        ):
        super(OthersEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            # modular_training=modular_training,
            # training_policy=training_policy,
            # loss_weight=loss_weight,
            # num_module=num_module,
            # modules_update=modules_update,
            # batchnorm=batchnorm,
            # alpha=alpha,
            # mad_time_update=mad_time_update,
            # temperature=temperature,
        )
        a=1
        num_module = 4

    def get_main_model(self):
        return self.main_model
    
    def entropy_prediction(self, img):
        x = self.extract_feat(img)

        entr, conf = self.decode_head.calculate_entropy(x)

        return {f"confidence": conf, "entropy": entr}


