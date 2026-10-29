"""Validation-only confidence/NMS sweep using a fixed trained YOLO checkpoint."""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent/'analysis_tools'))
from analyze_dental_candidates import overlaps


def measure(predictions, coco, nms):
    annotations = defaultdict(list)
    extra_ids = {c['id'] for c in coco['categories'] if c['name'] == 'extra_tooth'}
    for a in coco['annotations']:
        if not a.get('iscrowd', 0) and not a.get('ignore', False):
            annotations[a['image_id']].append(a)
    output = []
    for score in [.001, .005, .01, .025, .05, .1, .25]:
        for threshold in [.5, .75]:
            total = matched = extra_total = extra_matched = 0
            counts, misses = [], []
            for image, p in predictions:
                gt = annotations[image['id']]
                gt_boxes = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0]+a['bbox'][2], a['bbox'][1]+a['bbox'][3]] for a in gt]).reshape(-1, 4)
                selected = p['bboxes'][p['scores'] >= score]
                ious = overlaps(gt_boxes, selected)
                valid = ious >= threshold
                row, col = linear_sum_assignment(-(valid*(len(gt)+1)+ious))
                hit = set(row[valid[row, col]].tolist())
                total += len(gt)
                matched += len(hit)
                extra_total += sum(a['category_id'] in extra_ids for a in gt)
                extra_matched += sum(gt[i]['category_id'] in extra_ids for i in hit)
                misses.extend(dict(image_id=image['id'], annotation_id=a['id']) for i, a in enumerate(gt)
                              if a['category_id'] in extra_ids and i not in hit)
                counts.append(len(selected))
            output.append(dict(nms_iou=nms, score_threshold=score, matching_iou=threshold,
                gt=total, matched=matched, recall=matched/total,
                extra_gt=extra_total, extra_matched=extra_matched,
                mean_candidates=float(np.mean(counts)), p95_candidates=float(np.percentile(counts,95)),
                max_candidates=int(max(counts)), unmatched_per_image=(sum(counts)-matched)/len(counts),
                missed_extra=misses))
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--nms-grid', type=float, nargs='+', default=[.6, .75, .85])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    coco = json.loads((args.data_root/'fdi53/val.json').read_text())
    model = YOLO(str(args.checkpoint))
    summary = []
    for nms in args.nms_grid:
        predictions = []
        for image in coco['images']:
            # Explicit single-image calls avoid list-as-batch behavior.
            r = model.predict(source=str(args.data_root/image['file_name']), device=0,
                              imgsz=args.imgsz, rect=False, conf=.001, iou=nms,
                              max_det=500, verbose=False)[0]
            assert r.orig_shape == (image['height'], image['width'])
            predictions.append((image, dict(bboxes=r.boxes.xyxy.cpu().numpy(), scores=r.boxes.conf.cpu().numpy())))
        with (args.output/f'val_nms_{nms}.pkl').open('wb') as stream:
            pickle.dump(predictions, stream)
        rows = measure(predictions, coco, nms)
        summary.extend(rows)
        print('NMS_DONE',nms,'ceiling_hits',sum(len(p['scores'])==500 for _,p in predictions), flush=True)
        (args.output/'results.json').write_text(json.dumps(summary, indent=2))
    (args.output/'protocol.json').write_text(json.dumps(dict(split='val', checkpoint=str(args.checkpoint),
        imgsz=args.imgsz, batch=1, score_floor=.001, max_det=500, nms_grid=args.nms_grid,
        note='Post-NMS candidate availability only; no claim about pre-NMS recall or external performance'), indent=2))


if __name__ == '__main__':
    main()
