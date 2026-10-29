"""Offline, reversible geometric grouping; not a learned tooth instance model."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from analyze_dental_candidates import analyze, overlaps


def group_candidates(pred, threshold):
    boxes = np.asarray(pred['bboxes'])
    scores = np.asarray(pred['scores'])
    pending = np.argsort(-scores, kind='stable')
    groups = []
    while len(pending):
        leader = int(pending[0])
        similarity = overlaps(boxes[leader:leader+1], boxes[pending])[0]
        mask = similarity >= threshold
        mask[0] = True
        groups.append((leader, pending[mask].tolist()))
        pending = pending[~mask]
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions-root', type=Path, required=True)
    p.add_argument('--datasets-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    reports = []
    previews = []
    for path in sorted(args.predictions_root.glob('*/eval/*/predictions.pkl')):
        experiment = path.parents[2].name
        dataset = experiment.rsplit('_', 1)[0]
        coco = json.loads((args.datasets_root/dataset/'fdi53'/f'{path.parent.name}.json').read_text())
        names = [c['name'] for c in sorted(coco['categories'], key=lambda c: c['id'])]
        images = {image['id']: image for image in coco['images']}
        with path.open('rb') as stream:
            predictions = pickle.load(stream)
        for threshold in [.75, .85, .95]:
            grouped = []
            for image in predictions:
                pred = image['pred_instances']
                groups = group_candidates(pred, threshold)
                leaders = np.array([g[0] for g in groups], dtype=int)
                grouped.append(dict(img_id=image['img_id'], pred_instances=dict(
                    bboxes=np.asarray(pred['bboxes'])[leaders],
                    scores=np.asarray(pred['scores'])[leaders],
                    proposal_ids=np.arange(len(leaders)))))
                if threshold == .85 and path.parent.name == 'val' and len(previews) < 3:
                    entities = []
                    for leader, members in groups:
                        probs = np.asarray(pred['all_class_probs'])[leader, :-1]
                        top = np.argsort(-probs)[:3]
                        entities.append(dict(bbox=np.asarray(pred['bboxes'])[leader].tolist(),
                            score=float(pred['scores'][leader]),
                            layer='primary' if pred['scores'][leader] >= .1 else 'review',
                            class_hypotheses=[dict(label=int(k), category=names[k], probability=float(probs[k])) for k in top],
                            raw_candidate_indices=members))
                    previews.append(dict(experiment=experiment, img_id=image['img_id'],
                        image_size=dict(width=images[image['img_id']]['width'], height=images[image['img_id']]['height']),
                        coordinates='original_image_xyxy', grouping_is_provisional=True,
                        raw_artifact=str(path.relative_to(args.predictions_root)),
                        entities=entities))
            rows = [r for r in analyze(grouped, coco) if r['mode'] == 'all_candidates']
            reports.append(dict(experiment=experiment, split=path.parent.name,
                                grouping_iou=threshold, results=rows))
        print(experiment, path.parent.name, 'OK', flush=True)
    assert reports
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2))
    args.output.with_name('interface_preview.json').write_text(json.dumps(previews, indent=2))


if __name__ == '__main__':
    main()
