"""Analyze a trusted prediction artifact against original FDI annotations."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from analyze_dental_candidates import analyze
from validate_dental_interface import group_candidates


def evaluate(predictions, coco):
    groups = []
    for image in predictions:
        p = image['pred_instances']
        leaders = np.array([leader for leader, _ in group_candidates(p, .85)], dtype=int)
        groups.append(dict(img_id=image['img_id'], pred_instances=dict(
            bboxes=np.asarray(p['bboxes'])[leaders], scores=np.asarray(p['scores'])[leaders],
            proposal_ids=np.arange(len(leaders)))))
    return dict(raw=analyze(predictions, coco), grouped_iou_085=analyze(groups, coco))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    with args.predictions.open('rb') as stream:
        predictions = pickle.load(stream)
    results = evaluate(predictions, json.loads(args.annotations.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
