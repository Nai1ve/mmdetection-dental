"""Check durable artifacts before a structure controller advances to the next job."""
import argparse
import csv
import json
import math
from pathlib import Path


def validate(work, epochs):
    rows = list(csv.DictReader((work / 'results.csv').open()))
    assert len(rows) == epochs and float(rows[-1]['epoch']) == epochs
    assert all(math.isfinite(float(v)) for row in rows for k, v in row.items()
               if k.strip().startswith('train/'))
    assert (work / 'weights/best.pt').is_file()
    config = json.loads((work / 'structure_config.json').read_text())['options']
    for filename in ['assignment_audit.jsonl', 'quality_audit.jsonl']:
        audit = [json.loads(line) for line in (work / filename).read_text().splitlines()]
        assert len(audit) == epochs and [r['epoch'] for r in audit] == list(range(1, epochs + 1))
        if filename == 'quality_audit.jsonl':
            assert all(math.isfinite(row['quality_bce_mean']) for row in audit)
            if config['quality']:
                assert all(row['batches'] > 0 for row in audit)
        else:
            assert all(abs(row['weighted_positive_mass'] - row['positive_anchors']) <=
                       max(1., row['positive_anchors'] * 1e-5) for row in audit)
    for filename in ['evaluation_summary.json', 'evaluation_summary_nms085.json']:
        reports = json.loads((work / filename).read_text())
        assert len(reports) == 3
        assert {r['split'] for r in reports} == {'val', 'internal_test', 'external_test'}
        assert all(r['coco_stats'] is not None for r in reports)
        if config['quality']:
            folder = 'eval_nms085' if 'nms085' in filename else 'eval'
            for split in ['val', 'internal_test', 'external_test']:
                predictions = json.loads((work / folder / split / 'detections.json').read_text())
                assert all(0 <= r['localization_quality'] <= 1 for r in predictions)
    print('COMPLETE', work)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('work', type=Path)
    parser.add_argument('--epochs', type=int, default=24)
    args = parser.parse_args()
    validate(args.work, args.epochs)
