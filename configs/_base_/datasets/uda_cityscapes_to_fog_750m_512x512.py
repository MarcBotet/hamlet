# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='CityscapesDataset',
            data_root='/data/datasets/Cityscapes/',
            img_dir='leftImg8bit/train',
            ann_dir='data/Cityscapes/gtFine/train',
            pipeline=cityscapes_train_pipeline),
        target=dict(
            type='CityscapesDataset',
            domain=25,
            gt_shape=(1024, 2048, 3),
            data_root='/data/datasets/weather_datasets/weather_cityscapes/',
            img_dir='leftImg8bit/train/fog/750m/',
            ann_dir='data/Cityscapes/gtFine/train',
            pipeline=cityscapes_train_pipeline)),
    val=dict(
        type='CityscapesDataset',
        name='750m',
        gt_shape=(1024, 2048, 3),
        data_root='data/',
        img_dir='val/fog/750m/',
        ann_dir='data/Cityscapes/gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type='CityscapesDataset',
        name='750m',
        gt_shape=(1024, 2048, 3),
        data_root='data/',
        img_dir='val/fog/750m/',
        ann_dir='data/Cityscapes/gtFine/val',
        pipeline=test_pipeline))