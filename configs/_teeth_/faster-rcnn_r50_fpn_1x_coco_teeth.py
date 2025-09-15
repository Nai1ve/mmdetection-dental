# 新配置继承了基本配置，并做了必要的修改
_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
dataset_type = 'CocoDataset'
# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(num_classes=48)))

# 修改数据集相关配置
data_root = '../dataset/coco/crop_child/'
metainfo = {
    'classes': ('11','12','13','14','15','16','17',
                '21','22','23','24','25','26','27',
                '31','32','33','34','35','36','37',
                '41','42','43','44','45','46','47',
                '51','52','53','54','55',
                '61','62','63','64','65',
                '71','72','73','74','75',
                '81','82','83','84','85'
                ),
    'palette': [
        (230, 50, 80), (60, 180, 75), (0, 130, 200), (245, 150, 0), (145, 30, 180),
    (70, 240, 120), (220, 190, 40), (170, 110, 80), (30, 200, 220), (255, 0, 100),
    (40, 90, 160), (210, 80, 150), (100, 200, 0), (180, 0, 90), (50, 160, 200),
    (255, 120, 60), (10, 140, 70), (200, 50, 120), (80, 180, 220), (240, 60, 180),
    (120, 200, 50), (160, 30, 150), (20, 220, 160), (255, 90, 30), (70, 130, 210),
    (0, 180, 240), (220, 140, 100), (50, 70, 190), (180, 200, 40), (130, 0, 120),
    (200, 180, 0), (90, 50, 140), (30, 240, 100), (255, 70, 150), (110, 160, 60),
    (0, 100, 220), (240, 200, 80), (150, 80, 170), (70, 0, 160), (255, 160, 20),
    (40, 200, 140), (190, 60, 110), (80, 220, 180), (230, 0, 70), (120, 120, 200),
    (60, 150, 30), (210, 100, 180), (10, 80, 130)
    ]
}

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    #dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
        dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        # 对应 hsv_s: 0.7
        # MMDetection中的饱和度调整范围，YOLO的0.7gain可以近似为(1-0.7, 1+0.7)
        saturation_range=(0.3, 1.7), 
        # 对应 hsv_v: 0.4
        # MMDetection中通常用 brightness_delta, contrast_range, hue_delta 等来近似
        # MMDetection的hue_delta是 [-hue_delta, hue_delta]，
        # 这里为0，因为YOLO参数中hsv_h为0
        hue_delta=0,
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo = metainfo,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='preprocessing_images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
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
        metainfo = metainfo,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='preprocessing_images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric=['bbox'],
    format_only=False,
    classwise=True,
    backend_args=backend_args)
test_evaluator = val_evaluator