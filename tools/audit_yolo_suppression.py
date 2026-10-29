"""Replay NMS from the same forward pass and audit original extra-tooth GT."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops

sys.path.insert(0, str(Path(__file__).parent))
from validate_yolo_operating_points import measure
sys.path.insert(0, str(Path(__file__).parent/'analysis_tools'))
from analyze_dental_candidates import overlaps


class AuditPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs, **kwargs):
        raw = (preds[0] if isinstance(preds, (tuple, list)) else preds).detach().clone()
        assert raw.shape[0] == 1 and raw.shape[1] == 5, 'Expected batch-one, single-class YOLO11'
        network_boxes = ops.xywh2xyxy(raw[0, :4].T)
        original_boxes = ops.scale_boxes(img.shape[2:], network_boxes.clone(), orig_imgs[0].shape[:2])
        self.audit = dict(network_boxes=network_boxes, boxes=original_boxes, scores=raw[0, 4])
        return super().postprocess(preds, img, orig_imgs, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    coco = json.loads((args.data_root/'fdi53/val.json').read_text())
    extra = {c['id'] for c in coco['categories'] if c['name']=='extra_tooth'}
    gt = defaultdict(list)
    for a in coco['annotations']:
        gt[a['image_id']].append(a)
    predictions = {threshold: [] for threshold in [.6,.75,.85]}
    cases = []
    model = YOLO(str(args.checkpoint))
    for image in coco['images']:
        native = model.predict(source=str(args.data_root/image['file_name']), predictor=AuditPredictor,
            device=0, imgsz=960, rect=False, conf=.001, iou=.6, max_det=500, verbose=False)[0]
        raw = model.predictor.audit
        scores = raw['scores']
        selected = torch.where(scores > .001)[0]
        keeps = {}
        for threshold in predictions:
            keep = selected[nms(raw['network_boxes'][selected], scores[selected], threshold)[:500]]
            keeps[threshold] = keep.cpu().numpy()
            boxes = raw['boxes'][keep].cpu().numpy()
            pred_scores = scores[keep].cpu().numpy()
            if threshold == .6:
                np.testing.assert_allclose(boxes, native.boxes.xyxy.cpu().numpy(), atol=1e-3)
                np.testing.assert_allclose(pred_scores, native.boxes.conf.cpu().numpy(), atol=1e-6)
            predictions[threshold].append((image, dict(bboxes=boxes, scores=pred_scores)))
        boxes = raw['boxes'].cpu().numpy()
        score_array = scores.cpu().numpy()
        for ann in gt[image['id']]:
            if ann['category_id'] not in extra:
                continue
            x,y,w,h = ann['bbox']
            ious = overlaps(np.array([[x,y,x+w,y+h]]), boxes)[0]
            eligible = np.where((ious >= .5) & (score_array >= .01))[0]
            candidate = int(eligible[np.argmax(score_array[eligible])]) if len(eligible) else None
            info = dict(image_id=image['id'],file_name=image['file_name'],annotation_id=ann['id'],
                        gt_xyxy=[x,y,x+w,y+h], best_raw_iou=float(ious.max()),
                        max_score_iou05=float(score_array[ious>=.5].max()) if np.any(ious>=.5) else None,
                        max_score_iou075=float(score_array[ious>=.75].max()) if np.any(ious>=.75) else None,
                        highest_score_eligible_candidate=candidate, nms={})
            for threshold, keep in keeps.items():
                kept = keep[score_array[keep]>=.01]
                row = dict(available_iou05=int((ious[kept]>=.5).sum()),
                           available_iou075=int((ious[kept]>=.75).sum()))
                if candidate is not None:
                    row['candidate_survived'] = bool(candidate in keep)
                    higher = kept[score_array[kept] >= score_array[candidate]]
                    blocker_iou = overlaps(raw['network_boxes'][candidate:candidate+1].cpu().numpy(),
                                           raw['network_boxes'][higher].cpu().numpy())[0]
                    blockers = higher[blocker_iou > threshold]
                    row['higher_score_overlapping_survivors'] = [dict(raw_index=int(k),
                        box=boxes[k].tolist(),score=float(score_array[k]),iou_with_gt=float(ious[k])) for k in blockers[:3] if k != candidate]
                info['nms'][str(threshold)] = row
            best = int(ious.argmax())
            info['best_raw_box'] = boxes[best].tolist()
            info['best_raw_box_score'] = float(score_array[best])
            if candidate is not None:
                info['candidate_box'] = boxes[candidate].tolist()
                info['candidate_score'] = float(score_array[candidate])
            cases.append(info)
    results = [row for threshold,preds in predictions.items() for row in measure(preds,coco,threshold)]
    (args.output/'results.json').write_text(json.dumps(results,indent=2))
    (args.output/'extra_cases.json').write_text(json.dumps(cases,indent=2))
    (args.output/'verification.json').write_text(json.dumps(dict(
        images=len(coco['images']), extra_instances=len(cases), native_replay_verified=True,
        note='All NMS settings share identical forward-pass predictions; raw geometric boxes are not valid detections without score/one-to-one checks'),indent=2))
    print('AUDIT_OK',len(coco['images']),len(cases),flush=True)


if __name__ == '__main__':
    main()
