"""Evaluate trusted local DumpDetResults pickles with class-agnostic matching."""
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def overlaps(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(rb - lt, 0).prod(2)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-10)


def analyze(predictions, coco):
    gt = defaultdict(list)
    extra_ids = {c['id'] for c in coco['categories'] if c['name'] == 'extra_tooth'}
    for ann in coco['annotations']:
        if not ann.get('iscrowd', 0) and not ann.get('ignore', False):
            x, y, w, h = ann['bbox']
            gt[ann['image_id']].append(([x, y, x+w, y+h], ann['category_id'] in extra_ids))
    expected = {x['id'] for x in coco['images']}
    assert {p['img_id'] for p in predictions} == expected
    assert len(predictions) == len(expected)
    totals = defaultdict(lambda: np.zeros(6, dtype=float))
    for item in predictions:
        entries = gt[item['img_id']]
        boxes = np.array([a[0] for a in entries], dtype=float).reshape(-1, 4)
        extra = np.array([a[1] for a in entries], dtype=bool)
        p = item['pred_instances']
        scores = np.asarray(p['scores'])
        order = np.argsort(-scores, kind='stable')
        proposal_top1 = []
        seen = set()
        for i in order:
            pid = int(p['proposal_ids'][i])
            if pid not in seen:
                seen.add(pid)
                proposal_top1.append(i)
        for mode, indices in [('all_candidates', order), ('proposal_top1', np.array(proposal_top1, dtype=int))]:
            for score in [0.01, 0.05, 0.1, 0.25, 0.5]:
                selected = indices[scores[indices] >= score]
                ious = overlaps(boxes, np.asarray(p['bboxes'])[selected])
                for threshold in [0.4, 0.5, 0.75]:
                    valid = ious >= threshold
                    # Cardinality dominates IoU, so one large box cannot match two teeth.
                    rows, cols = linear_sum_assignment(-(valid * (len(boxes)+1) + ious))
                    matched = rows[valid[rows, cols]]
                    totals[mode, score, threshold] += [len(boxes), len(matched), extra.sum(), extra[matched].sum(), len(selected), 1]
    rows = []
    for (mode, score, threshold), (n, tp, ne, te, npred, images) in totals.items():
        rows.append(dict(mode=mode, score_threshold=score, iou_threshold=threshold,
                         gt=int(n), matched=int(tp), fn=int(n-tp), recall=tp/n if n else None,
                         extra_gt=int(ne), extra_matched=int(te), extra_recall=te/ne if ne else None,
                         candidates_per_image=npred/images, unmatched_candidates_per_image=(npred-tp)/images))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions-root', type=Path, required=True)
    parser.add_argument('--datasets-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = []
    for path in sorted(args.predictions_root.glob('*/eval/*/predictions.pkl')):
        experiment = path.parents[2].name
        dataset = experiment.rsplit('_', 1)[0]
        split = path.parent.name
        coco = json.loads((args.datasets_root / dataset / 'fdi53' / f'{split}.json').read_text())
        with path.open('rb') as stream:
            predictions = pickle.load(stream)  # Only use artifacts from our trusted server.
        rows = analyze(predictions, coco)
        output.append(dict(experiment=experiment, split=split, results=rows))
        print(experiment, split, 'OK', flush=True)
        del predictions
    assert output, 'No prediction files found'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
