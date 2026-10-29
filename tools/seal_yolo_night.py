"""Create a compact, durable statistics bundle before authorized shutdown."""
import argparse
import csv
import hashlib
import json
import math
import tarfile
from pathlib import Path
import yaml

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--failures',type=int,required=True);args=p.parse_args()
base=Path('/root/autodl-tmp'); rows=list(csv.DictReader((args.run/'status.csv').open())); summary=[]
assert len(rows)==9, 'Expected eight training jobs plus Soft-NMS validation'
for row in rows:
    if row['status']!='0': continue
    work=Path(row['work_dir'])
    if row['variant']=='postprocess':continue
    training=list(csv.DictReader((work/'results.csv').open()))
    assert len(training)==24 and float(training[-1]['epoch'])==24
    assert all(math.isfinite(float(value)) for epoch in training for key,value in epoch.items() if key.strip().startswith('train/')), 'Non-finite training loss'
    config=yaml.safe_load((work/'args.yaml').read_text())
    assert config['batch']==8 and config['imgsz']==960 and config['seed']==int(row['seed']), 'Recipe drift'
    if row['variant']!='standard':
        audit=[json.loads(line) for line in (work/'assignment_audit.jsonl').read_text().splitlines()]
        assert len(audit)==24 and all(a['variant']==row['variant'] for a in audit)
        assert all(abs(a['weighted_positive_mass']-a['positive_anchors'])<=max(1.,a['positive_anchors']*1e-5) for a in audit), 'Foreground mass invariant failed'
    for suffix in ['', '_nms085']:
        path=work/f'evaluation_summary{suffix}.json'
        data=json.loads(path.read_text())
        summary.append(dict(job=row['job'],variant=row['variant'],seed=int(row['seed']),
            postprocessing=suffix or 'nms06',results=data))
(args.run/'batch_summary.json').write_text(json.dumps(summary,indent=2))
(args.run/'sealed_source_hashes.json').write_text(json.dumps({name:hashlib.sha256((Path('/root/mmdetection-dental/tools')/name).read_bytes()).hexdigest() for name in ['dental_yolo_ablation.py','run_modern_dental.py','run_yolo_night.sh','seal_yolo_night.py']},indent=2))
archive=args.run/'statistics_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    roots=[args.run]+sorted({Path(r['work_dir']) for r in rows})
    for root in roots:
        for file in root.rglob('*'):
            if file.is_file() and file.suffix in ['.json','.jsonl','.csv','.yaml','.log','.txt']:
                tar.add(file,arcname=str(file.relative_to(base)))
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
(args.run/'sealed.json').write_text(json.dumps(dict(failures=args.failures,status_rows=len(rows),
    evaluated_model_settings=len(summary),archive=str(archive),sha256=digest),indent=2))
print('SEALED',archive,digest,flush=True)
