_base_ = [
    '../_base_/models/bts.py', '../_base_/datasets/nyu.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    decode_head=dict(
        final_norm=False,
        min_depth=1e-3,
        max_depth=10,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    )

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    test=dict(
        data_root="data/track2/",
        split='test.txt'))