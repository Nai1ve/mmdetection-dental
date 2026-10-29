"""Matched high-recall Faster R-CNN baseline for the rebuilt private dataset."""
import os

_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
custom_imports = dict(imports=['mmdet.models.roi_heads.dental_candidate_roi_head'],
                      allow_failed_imports=False)
data_root = os.environ.get('DENTAL_DATA_ROOT', '/root/autodl-tmp/private_only').rstrip('/')+'/'
test_split = os.environ.get('DENTAL_TEST_SPLIT', 'external_test' if
    os.path.isfile(data_root+'fdi53/external_test.json') else 'external_huairou')
classes = tuple(str(n) for a,b in [(11,18),(21,28),(31,38),(41,48),
    (51,55),(61,65),(71,75),(81,85)] for n in range(a,b+1))+('extra_tooth',)
metainfo = dict(classes=classes)
model = dict(
    roi_head=dict(type='DentalCandidateRoIHead', bbox_head=dict(
        type='Shared2FCBBoxHead', num_classes=53)),
    train_cfg=dict(rpn_proposal=dict(nms_pre=3000, max_per_img=2000,
        nms=dict(type='nms',iou_threshold=0.8))),
    test_cfg=dict(rpn=dict(nms_pre=3000,max_per_img=2000,
        nms=dict(type='nms',iou_threshold=0.8)),
        rcnn=dict(score_thr=0.01,nms=dict(type='nms',iou_threshold=0.6),max_per_img=500)))
train_pipeline = [dict(type='LoadImageFromFile'),dict(type='LoadAnnotations',with_bbox=True),
    dict(type='Resize',scale=(1600,960),keep_ratio=True),dict(type='PackDetInputs')]
test_pipeline = [dict(type='LoadImageFromFile'),
    dict(type='Resize',scale=(1600,960),keep_ratio=True),
    dict(type='LoadAnnotations',with_bbox=True),
    dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))]
train_dataloader = dict(batch_size=2,num_workers=4,dataset=dict(
    data_root=data_root,ann_file='fdi53/train.json',data_prefix=dict(img=''),
    metainfo=metainfo,pipeline=train_pipeline))
val_dataloader = dict(batch_size=1,num_workers=2,dataset=dict(
    data_root=data_root,ann_file='fdi53/val.json',data_prefix=dict(img=''),
    metainfo=metainfo,pipeline=test_pipeline))
test_dataloader = dict(batch_size=1,num_workers=2,dataset=dict(
    data_root=data_root,ann_file=f'fdi53/{test_split}.json',data_prefix=dict(img=''),
    metainfo=metainfo,pipeline=test_pipeline))
val_evaluator = dict(ann_file=data_root+'fdi53/val.json',classwise=True)
test_evaluator = dict(ann_file=data_root+f'fdi53/{test_split}.json',classwise=True)
train_cfg = dict(type='EpochBasedTrainLoop',max_epochs=24,val_interval=1)
param_scheduler = [dict(type='LinearLR',start_factor=0.001,by_epoch=False,begin=0,end=500),
    dict(type='MultiStepLR',begin=0,end=24,by_epoch=True,milestones=[16,22],gamma=0.1)]
optim_wrapper = dict(optimizer=dict(lr=0.0025))
randomness = dict(seed=42,deterministic=False)
default_hooks = dict(checkpoint=dict(interval=1,max_keep_ckpts=1,
    save_best='coco/bbox_mAP',rule='greater'))
