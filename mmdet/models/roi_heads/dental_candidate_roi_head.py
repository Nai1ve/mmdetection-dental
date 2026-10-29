"""Shared inference export for the standard and CCSW ablation models."""
import torch
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.models.layers import multiclass_nms
from mmdet.structures.bbox import bbox2roi
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class DentalCandidateRoIHead(StandardRoIHead):
    def predict_bbox(self, x, batch_img_metas, rpn_results_list,
                     rcnn_test_cfg, rescale=False):
        proposals = [r.bboxes for r in rpn_results_list]
        counts = [len(p) for p in proposals]
        rois = bbox2roi(proposals)
        if not len(rois):
            results = super().predict_bbox(x, batch_img_metas, rpn_results_list,
                                           rcnn_test_cfg, rescale)
            if rcnn_test_cfg is not None:
                for result in results:
                    result.proposal_ids = rois.new_empty((0,), dtype=torch.long)
                    result.visual_features = rois.new_empty((0, self.bbox_head.fc_out_channels))
                    result.all_class_probs = rois.new_empty((0, self.bbox_head.num_classes+1))
                    result.foreground_scores = rois.new_empty((0,))
            return results
        out = self._bbox_forward(x, rois)
        head = self.bbox_head
        features = out['bbox_feats'].flatten(1)
        for fc in head.shared_fcs:
            features = head.relu(fc(features))
        parts = zip(rois.split(counts), out['cls_score'].split(counts),
                    out['bbox_pred'].split(counts), features.split(counts),
                    batch_img_metas)
        results = []
        for roi, logits, delta, descriptors, meta in parts:
            raw = head._predict_by_feat_single(
                roi, logits, delta, meta, rescale=rescale, rcnn_test_cfg=None)
            if rcnn_test_cfg is None:
                results.append(raw)
                continue
            boxes, labels, keep = multiclass_nms(
                raw.bboxes, raw.scores, rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img, return_inds=True)
            proposal_ids = keep // head.num_classes
            results.append(InstanceData(bboxes=boxes[:, :4], scores=boxes[:, 4],
                labels=labels, proposal_ids=proposal_ids,
                visual_features=descriptors[proposal_ids],
                all_class_probs=raw.scores[proposal_ids],
                foreground_scores=1-raw.scores[proposal_ids, -1]))
        return results
