# Obtained from: https://github.com/NVlabs/SegFormer
# Modifications: BN instead of SyncBN

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='ModularEncoderDecoder',
    pretrained=None,
    backbone=dict(type='IMTRv21_5', style='pytorch'),
    total_modules=4,
    decode_head=dict(
        type='IncrementalSegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
