"""Jaw-aware positive classification weighting; standard bbox regression."""
import torch
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor
from .convfc_bbox_head import Shared2FCBBoxHead


@MODELS.register_module()
class CCSWBBOXHead(Shared2FCBBoxHead):
    """Weak crown-direction prior from full-tooth boxes, not crown annotation.

    Inherits the standard two-output forward and prediction interfaces.
    With split=0.5 a GT-aligned proposal gets neutral weight 1.
    """
    def __init__(self, *args, crown_weight=1.5, root_weight=0.5,
                 crown_root_split_ratio=0.5, primary_teeth_indices=None,
                 upper_teeth_indices=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= root_weight <= 1 <= crown_weight:
            raise ValueError('Require 0 <= root_weight <= 1 <= crown_weight')
        if not 0 < crown_root_split_ratio < 1:
            raise ValueError('Split ratio must be in (0, 1)')
        if self.num_classes not in (48, 53) and (
                primary_teeth_indices is None or upper_teeth_indices is None):
            raise ValueError('Explicit tooth indices required for this ontology')
        start = 28 if self.num_classes == 48 else 32
        upper_count = 14 if self.num_classes == 48 else 16
        self.primary_teeth_indices = tuple(primary_teeth_indices or (start, start+19))
        self.upper_teeth_indices = tuple(upper_teeth_indices or
            (*range(upper_count), *range(start, start+10)))
        self.crown_weight = crown_weight
        self.root_weight = root_weight
        self.crown_root_split_ratio = crown_root_split_ratio

    def spatial_label_weights(self, rois, labels, label_weights, sampling_results):
        weights = label_weights.clone()
        positive = (labels >= 0) & (labels < self.num_classes)
        if not positive.any():
            return weights
        gt = torch.cat([get_box_tensor(r.pos_gt_bboxes) for r in sampling_results])
        boxes = rois[positive, 1:]
        pos_labels = labels[positive]
        if len(gt) != len(boxes):
            raise ValueError('Positive GT/proposal length mismatch')
        upper = torch.zeros_like(pos_labels, dtype=torch.bool)
        for idx in self.upper_teeth_indices:
            upper |= pos_labels == idx
        low, high = self.primary_teeth_indices
        selected = (pos_labels >= low) & (pos_labels <= high)
        position = ((boxes[:, 1]+boxes[:, 3])*.5-gt[:, 1]) / (
            gt[:, 3]-gt[:, 1]).clamp_min(1e-6)
        # Mirror the split for lower teeth; upper crowns point downwards.
        split = torch.where(upper, self.crown_root_split_ratio,
                            1-self.crown_root_split_ratio)
        signed = torch.where(upper, position-split, split-position)
        signed = (signed / max(self.crown_root_split_ratio,
                               1-self.crown_root_split_ratio)).clamp(-1, 1)
        factors = torch.where(signed >= 0,
            1+signed*(self.crown_weight-1), 1+signed*(1-self.root_weight))
        weights[positive] *= torch.where(selected, factors, torch.ones_like(factors))
        return weights

    def loss_and_target(self, cls_score, bbox_pred, rois, sampling_results,
                        rcnn_train_cfg, concat=True, reduction_override=None):
        if not concat:
            raise ValueError('CCSW requires concatenated targets')
        targets = self.get_targets(sampling_results, rcnn_train_cfg, concat=True)
        labels, label_weights, bbox_targets, bbox_weights = targets
        weighted = self.spatial_label_weights(rois, labels, label_weights, sampling_results)
        losses = super().loss(cls_score, bbox_pred, rois, labels, weighted,
                              bbox_targets, bbox_weights,
                              reduction_override=reduction_override)
        return dict(loss_bbox=losses, bbox_targets=targets)
