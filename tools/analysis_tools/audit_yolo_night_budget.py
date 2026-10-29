"""Replay archived COCO detections at fixed per-image budgets; no inference."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from analyze_dental_candidates import overlaps


def measure(coco, detections, budget):
    truth, predicted = defaultdict(list), defaultdict(list)
    extra_ids = {c['id'] for c in coco['categories'] if c['name'] == 'extra_tooth'}
    expected = {image['id'] for image in coco['images']}
    for ann in coco['annotations']:
        if not ann.get('iscrowd', 0) and not ann.get('ignore', False):
            truth[ann['image_id']].append(ann)
    for detection in detections:
        assert detection['image_id'] in expected
        if detection['score'] >= .01:
            predicted[detection['image_id']].append(detection)
    totals = {t: dict(gt=0, matched=0, extra_gt=0, extra_matched=0,
                      candidates=0, extra_missed_ids=[]) for t in [.5, .75]}
    for image_id in sorted(expected):
        gt = truth[image_id]
        candidates = sorted(predicted[image_id], key=lambda x: -x['score'])
        if budget is not None:
            candidates = candidates[:budget]

        def xyxy(rows):
            return np.array([[x['bbox'][0], x['bbox'][1],
                              x['bbox'][0] + x['bbox'][2],
                              x['bbox'][1] + x['bbox'][3]] for x in rows],
                            dtype=float).reshape(-1, 4)

        ious = overlaps(xyxy(gt), xyxy(candidates))
        for threshold, total in totals.items():
            valid = ious >= threshold
            # Identical cardinality-first, IoU-second matching to archived evaluation.
            rows, cols = linear_sum_assignment(-(valid * (len(gt) + 1) + ious))
            matched = set(rows[valid[rows, cols]].tolist())
            extra = {i for i, ann in enumerate(gt) if ann['category_id'] in extra_ids}
            total['gt'] += len(gt)
            total['matched'] += len(matched)
            total['extra_gt'] += len(extra)
            total['extra_matched'] += len(matched & extra)
            total['candidates'] += len(candidates)
            total['extra_missed_ids'].extend(gt[i]['id'] for i in sorted(extra - matched))
    for total in totals.values():
        total['recall'] = total['matched'] / total['gt'] if total['gt'] else None
        total['candidates_per_image'] = total['candidates'] / len(expected)
        total['unmatched_per_image'] = (total['candidates'] - total['matched']) / len(expected)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.raw / 'experiments/yolo_ablation_20261030/batch_summary.json').read_text())
    result = dict(protocol='conf>=0.01; score-sorted top K per image; cardinality-first matching',
                  test_metrics=[], validation_budgets=[])
    for run in summary:
        identity = {k: run[k] for k in ['job', 'variant', 'seed', 'postprocessing']}
        for split in run['results']:
            rows = [r for r in split['candidate_metrics']['raw']
                    if r['mode'] == 'all_candidates' and r['score_threshold'] == .01
                    and r['iou_threshold'] in [.5, .75]]
            result['test_metrics'].append(dict(**identity, split=split['split'],
                                              coco_ap=split['coco_stats'][0], metrics=rows))
        folder = 'eval' if run['postprocessing'] == 'nms06' else 'eval_nms085'
        path = args.raw / 'work_dirs' / ('yolo_night_' + run['job']) / folder / 'val/detections.json'
        coco = json.loads((args.dataset / 'fdi53/val.json').read_text())
        detections = json.loads(path.read_text())
        for budget in [None, 40, 50, 60]:
            measured = measure(coco, detections, budget)
            if budget is None:
                archived = next(x for x in run['results'] if x['split'] == 'val')
                for threshold, value in measured.items():
                    old = next(r for r in archived['candidate_metrics']['raw']
                               if r['mode'] == 'all_candidates' and r['score_threshold'] == .01
                               and r['iou_threshold'] == threshold)
                    assert value['matched'] == old['matched'], (identity, threshold)
                    assert value['extra_matched'] == old['extra_matched'], (identity, threshold)
                    assert abs(value['candidates_per_image'] - old['candidates_per_image']) < 1e-8
            result['validation_budgets'].append(dict(**identity, budget=budget, metrics=measured))
        print(run['job'], run['postprocessing'], 'verified', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
