"""Seal completed structure runs without changing the live training controller."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

from validate_yolo_structure_run import validate


def seal(run, expected):
    rows = list(csv.DictReader((run / 'status.csv').open()))
    successes = [r for r in rows if r['state'] == 'SUCCESS']
    assert len(successes) == expected and len({r['job'] for r in successes}) == expected
    assert not any(r['state'] == 'FAILED' for r in rows)
    roots = [run]
    reports = []
    for row in successes:
        work = Path(row['work_dir'])
        validate(work, 24)
        roots.append(work)
        reports.append(dict(job=row['job'], seed=int(row['seed']),
            structure=json.loads((work / 'structure_config.json').read_text()),
            nms06=json.loads((work / 'evaluation_summary.json').read_text()),
            nms085=json.loads((work / 'evaluation_summary_nms085.json').read_text())))
    (run / 'batch_summary.json').write_text(json.dumps(reports, indent=2))
    base = run.parent.parent
    archive = run / 'statistics_bundle.tar.gz'
    temporary = run / 'statistics_bundle.tar.gz.partial'
    with tarfile.open(temporary, 'w:gz') as tar:
        for root in roots:
            for path in sorted(root.rglob('*')):
                if path.is_file() and path.suffix in {'.json', '.jsonl', '.csv', '.yaml', '.log', '.txt', '.pid'}:
                    # Do not bundle old seal records, potentially linking a newer archive to an old hash.
                    if path.name not in {'sealed.json', 'watcher_error.json'}:
                        tar.add(path, arcname=str(path.relative_to(base)))
    temporary.replace(archive)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    marker = run / 'sealed.json.tmp'
    marker.write_text(json.dumps(dict(failures=0, successful_jobs=expected,
        evaluated_model_settings=expected * 2, evaluated_subsets=expected * 6,
        archive=str(archive), sha256=digest.hexdigest()), indent=2))
    marker.replace(run / 'sealed.json')
    print('SEALED', archive, digest.hexdigest(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--expected', type=int, default=12)
    args = parser.parse_args()
    (args.run / 'watcher.pid').write_text(str(os.getpid()))
    try:
        while not (args.run / 'completed.txt').exists():
            rows = list(csv.DictReader((args.run / 'status.csv').open()))
            assert not any(r['state'] == 'FAILED' for r in rows), 'Controller recorded a failed job'
            pid = int((args.run / 'runner.pid').read_text())
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                if (args.run / 'completed.txt').exists():
                    break
                raise RuntimeError('Controller exited without completion')
            time.sleep(30)
        seal(args.run, args.expected)
    except Exception as exc:
        (args.run / 'watcher_error.json').write_text(json.dumps(dict(error=str(exc))))
        raise


if __name__ == '__main__':
    main()
