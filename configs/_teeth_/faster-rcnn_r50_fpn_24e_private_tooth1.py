"""Single-class task-definition control; retain the baseline training recipe."""
import os

_base_ = './faster-rcnn_r50_fpn_24e_private_baseline.py'
data_root = os.environ.get('DENTAL_DATA_ROOT', '/root/autodl-tmp/datasets/private_only').rstrip('/')+'/'
test_split = os.environ.get('DENTAL_TEST_SPLIT', 'external_test')
classes = ('tooth',)
metainfo = dict(classes=classes)
model = dict(roi_head=dict(bbox_head=dict(num_classes=1, reg_class_agnostic=True)))
train_dataloader = dict(dataset=dict(ann_file='tooth1/train.json', metainfo=metainfo))
val_dataloader = dict(dataset=dict(ann_file='tooth1/val.json', metainfo=metainfo))
test_dataloader = dict(dataset=dict(ann_file=f'tooth1/{test_split}.json', metainfo=metainfo))
val_evaluator = dict(ann_file=data_root+'tooth1/val.json')
test_evaluator = dict(ann_file=data_root+f'tooth1/{test_split}.json')
