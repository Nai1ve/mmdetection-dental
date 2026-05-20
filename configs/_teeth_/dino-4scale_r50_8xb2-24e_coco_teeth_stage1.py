_base_ = '../dino/dino-4scale_r50_8xb2-24e_coco.py'

data_root = '/root/autodl-tmp/private_coco_53/'
dataset_type = 'CocoDataset'
backend_args = None

# Keep the full arch context. Horizontal flip and random crop can break
# left-right FDI semantics and hurt slot-level recall.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[
            (480, 1333),
            (512, 1333),
            (544, 1333),
            (576, 1333),
            (608, 1333),
            (640, 1333),
            (672, 1333),
            (704, 1333),
            (736, 1333),
            (768, 1333),
            (800, 1333),
        ],
        keep_ratio=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

metainfo = dict(
    classes=(
        '11', '12', '13', '14', '15', '16', '17', '18', '21', '22', '23',
        '24', '25', '26', '27', '28', '31', '32', '33', '34', '35', '36',
        '37', '38', '41', '42', '43', '44', '45', '46', '47', '48', '51',
        '52', '53', '54', '55', '61', '62', '63', '64', '65', '71', '72',
        '73', '74', '75', '81', '82', '83', '84', '85', 'extra_tooth'))

model = dict(
    backbone=dict(frozen_stages=-1),
    positional_encoding=dict(offset=-0.5, temperature=10000),
    bbox_head=dict(
        num_classes=53,
        loss_cls=dict(loss_weight=2.0)),
    dn_cfg=dict(group_cfg=dict(num_dn_queries=300)),
    test_cfg=dict(max_per_img=500))

optim_wrapper = dict(
    optimizer=dict(lr=0.0002),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
        }))

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img=''),
        ann_file='train.json',
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img=''),
        ann_file='val.json',
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img=''),
        ann_file='internal_test.json',
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'internal_test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, save_best='coco/bbox_mAP_50'))

load_from = (
    'https://download.openmmlab.com/mmdetection/v3.0/dino/'
    'dino-4scale_r50_improved_8xb2-12e_coco/'
    'dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth')

# Persist logs and checkpoints outside the repo on AutoDL.
work_dir = '/root/autodl-tmp/work_dirs/stage1_dino_private_53'
