"""Same training recipe as baseline; only classification weighting changes."""
_base_ = './faster-rcnn_r50_fpn_24e_private_baseline.py'
model = dict(roi_head=dict(bbox_head=dict(type='CCSWBBOXHead',
    crown_weight=1.5,root_weight=0.5,crown_root_split_ratio=0.5,
    primary_teeth_indices=(32,51),
    upper_teeth_indices=tuple(range(16))+tuple(range(32,42)))))
