norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
conv_stem_norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoder',
    pretrained=
    'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
    backbone=dict(
        type='DepthFormerSwin',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        conv_norm_cfg=dict(type='BN', requires_grad=True),
        depth=50,
        num_stages=0),
    decode_head=dict(
        type='DenseDepthHead',
        in_channels=[64, 96, 192, 384, 768],
        up_sample_channels=[64, 96, 192, 384, 768],
        channels=64,
        align_corners=True,
        loss_decode=dict(type='SigLoss', valid_mask=True, loss_weight=1.0),
        act_cfg=dict(type='LeakyReLU', inplace=True),
        min_depth=0.001,
        max_depth=10),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    neck=dict(
        type='HAHIHeteroNeck',
        positional_encoding=dict(type='SinePositionalEncoding', num_feats=128),
        in_channels=[64, 96, 192, 384, 768],
        out_channels=[64, 96, 192, 384, 768],
        embedding_dim=256,
        scales=[1, 1, 1, 1, 1]))
dataset_type = 'NYUDataset'
data_root = 'data/nyu/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (416, 544)
train_pipeline = [
    dict(type='LoadImageFromFile_v2'),
    dict(type='DepthLoadAnnotations_v2'),
    dict(type='resize_cutmix'),
    dict(type='NYUCrop', depth=True),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(416, 544)),
    dict(
        type='ColorAug',
        prob=0.5,
        gamma_range=[0.9, 1.1],
        brightness_range=[0.75, 1.25],
        color_range=[0.9, 1.1]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'depth_gt'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'img_norm_cfg', 'cam_intrinsic'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(480, 640),
        flip=True,
        flip_direction='horizontal',
        transforms=[
            dict(type='RandomFlip', direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg', 'cam_intrinsic'))
        ])
]
eval_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', prob=0.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'img_norm_cfg', 'cam_intrinsic'))
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='NYUDataset',
        data_root='data/nyu/',
        depth_scale=1000,
        split='nyu_train.txt',
        pipeline=[
            dict(type='LoadImageFromFile_v2'),
            dict(type='DepthLoadAnnotations_v2'),
            dict(type='resize_cutmix'),
            dict(type='NYUCrop', depth=True),
            dict(type='RandomRotate', prob=0.5, degree=2.5),
            dict(type='RandomFlip', prob=0.5),
            dict(type='RandomCrop', crop_size=(416, 544)),
            dict(
                type='ColorAug',
                prob=0.5,
                gamma_range=[0.9, 1.1],
                brightness_range=[0.75, 1.25],
                color_range=[0.9, 1.1]),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'depth_gt'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg', 'cam_intrinsic'))
        ],
        garg_crop=False,
        eigen_crop=True,
        min_depth=0.001,
        max_depth=10),
    val=dict(
        type='NYUDataset',
        data_root='data/nyu/',
        depth_scale=1000,
        split='nyu_test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(480, 640),
                flip=True,
                flip_direction='horizontal',
                transforms=[
                    dict(type='RandomFlip', direction='horizontal'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                                   'img_shape', 'pad_shape', 'scale_factor',
                                   'flip', 'flip_direction', 'img_norm_cfg',
                                   'cam_intrinsic'))
                ])
        ],
        garg_crop=False,
        eigen_crop=True,
        min_depth=0.001,
        max_depth=10),
    test=dict(
        type='NYUDataset',
        data_root='data/track2/',
        depth_scale=1000,
        split='test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(480, 640),
                flip=True,
                flip_direction='horizontal',
                transforms=[
                    dict(type='RandomFlip', direction='horizontal'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                                   'img_shape', 'pad_shape', 'scale_factor',
                                   'flip', 'flip_direction', 'img_norm_cfg',
                                   'cam_intrinsic'))
                ])
        ],
        garg_crop=False,
        eigen_crop=True,
        min_depth=0.001,
        max_depth=10))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
max_lr = 1.5e-05
optimizer = dict(
    type='AdamW',
    lr=1.5e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=12800,
    warmup_ratio=0.001,
    min_lr_ratio=1e-08,
    by_epoch=False)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type='IterBasedRunner', max_iters=153600)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=1600)
evaluation = dict(
    by_epoch=False,
    start=0,
    interval=1600,
    pre_eval=True,
    rule='less',
    save_best='abs_rel',
    greater_keys=('a1', 'a2', 'a3'),
    less_keys=('abs_rel', 'rmse'))
work_dir = 'dir=work_dirs/2023-02-08-20-31-depthmix_test'
gpu_ids = range(0, 1)
