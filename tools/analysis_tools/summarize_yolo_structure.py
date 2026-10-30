"""Verify the structure statistics archive and summarize every seed and split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import tarfile


def summarize(root):
    sealed = json.loads((root / 'sealed.json').read_text())
    assert (sealed['failures'], sealed['successful_jobs'], sealed['evaluated_model_settings'],
            sealed['evaluated_subsets']) == (0, 12, 24, 72)
    archive = root / 'statistics_bundle.tar.gz'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == sealed['sha256'], 'Archive checksum mismatch'
    raw = root / 'raw'
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and '..' not in path.parts
            assert member.isfile() or member.isdir(), 'Links and special files forbidden'
        tar.extractall(raw, filter='data')
    run = raw / 'experiments/yolo_structure_core_20261030_0940'
    status = {r['job']: r for r in csv.DictReader((run / 'status.csv').open())}
    assert len(status) == 12 and all(r['state'] == 'SUCCESS' for r in status.values())
    runs = json.loads((run / 'batch_summary.json').read_text())
    assert len(runs) == 12
    rows = []
    for item in runs:
        method = item['job'].rsplit('_s', 1)[0]
        for nms in ['nms06', 'nms085']:
            assert {r['split'] for r in item[nms]} == {'val', 'internal_test', 'external_test'}
            for report in item[nms]:
                for mode in ['raw', 'grouped_iou_085']:
                    metrics = report['candidate_metrics'][mode]
                    selected = {m['iou_threshold']: m for m in metrics
                                if m['mode'] == 'all_candidates' and m['score_threshold'] == .01}
                    lo, hi = selected[.5], selected[.75]
                    rows.append(dict(method=method, seed=item['seed'], nms=nms, mode=mode,
                        split=report['split'], raw_coco_ap=report['coco_stats'][0], recall05=lo['recall'],
                        recall075=hi['recall'], extra05=lo['extra_matched'], extra075=hi['extra_matched'],
                        extra_gt=lo['extra_gt'], candidates=lo['candidates_per_image'],
                        parameters=item['structure']['total_parameters']))
    summary = []
    for split in ['val', 'internal_test', 'external_test']:
        for method in ['baseline', 'control', 'direction', 'direction_balance']:
            for nms in ['nms06', 'nms085']:
                for mode in ['raw', 'grouped_iou_085']:
                    group = sorted([r for r in rows if (r['method'], r['split'], r['nms'], r['mode']) ==
                                    (method, split, nms, mode)], key=lambda r: r['seed'])
                    assert [r['seed'] for r in group] == [42, 43, 44]
                    summary.append(dict(method=method, split=split, nms=nms, mode=mode,
                        means={k: statistics.mean(r[k] for r in group)
                               for k in ['raw_coco_ap', 'recall05', 'recall075', 'extra05', 'extra075', 'candidates']},
                        sample_sd={k: statistics.stdev(r[k] for r in group)
                                   for k in ['raw_coco_ap', 'recall05', 'recall075', 'extra05', 'extra075', 'candidates']},
                        extra05_by_seed=[r['extra05'] for r in group],
                        extra075_by_seed=[r['extra075'] for r in group],
                        extra_gt=group[0]['extra_gt'], parameters=group[0]['parameters']))
    result = dict(sha256=digest, completed=(run / 'completed.txt').read_text().strip(),
                  rows=rows, summary=summary)
    (root / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    for row in summary:
        if row['nms'] == 'nms085' and row['mode'] == 'raw':
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    summarize(parser.parse_args().root)
