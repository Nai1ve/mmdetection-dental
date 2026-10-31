#!/usr/bin/env bash
set -Eeuo pipefail
: "${RUN_ROOT:?Use a new data-disk experiment directory}"
PYTHON=${PYTHON:-/root/autodl-tmp/envs/ultralytics/bin/python}
CACHE_ROOT=${CACHE_ROOT:-/root/autodl-tmp/experiments/roi_recovery_matrix_20261031_1550}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets/private_only}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir "$RUN_ROOT"
echo $$ > "$RUN_ROOT/runner.pid"
git rev-parse HEAD > "$RUN_ROOT/commit.txt"
sha256sum tools/run_dental_roi.py tools/dental_roi_head.py tools/dental_roi_sampling.py tools/audit_roi_supervision.py > "$RUN_ROOT/source_sha256.txt"
printf 'job,state\n' > "$RUN_ROOT/status.csv"
trap 'printf "controller,FAILED\n" >> "$RUN_ROOT/status.csv"' ERR
# Reuse the sealed 7x7 cache: no new feature extraction or data split changes.
"$PYTHON" - "$CACHE_ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);sha=set()
for split,n in [('train',1148),('val',164)]:
    m=json.loads((r/f'cache_7_{split}/manifest.json').read_text())
    assert m['split']==split and m['images']==n and not m['partial_dataset'] and m['roi_size']==7
    sha.add(m['detector_sha256'])
assert len(sha)==1
PY
for split in train val; do
    "$PYTHON" tools/audit_roi_supervision.py --cache "$CACHE_ROOT/cache_7_$split" --output "$RUN_ROOT/supervision_$split.json" > "$RUN_ROOT/supervision_$split.log" 2>&1
done
for mode in single multi; do
    for policy in uniform stratified; do
        job="${mode}_${policy}"
        printf '%s,RUNNING\n' "$job" >> "$RUN_ROOT/status.csv"
        "$PYTHON" tools/run_dental_roi.py train --cache "$CACHE_ROOT/cache_7_train" --mode "$mode" --sampling "$policy" --epochs 6 --seed 42 --output "$RUN_ROOT/$job" > "$RUN_ROOT/$job.log" 2>&1
        for threshold in .25 .5; do
            out="$RUN_ROOT/${job}_val_${threshold}"
            "$PYTHON" tools/run_dental_roi.py eval --cache "$CACHE_ROOT/cache_7_val" --head "$RUN_ROOT/$job/last.pt" --data-root "$DATA_ROOT" --threshold "$threshold" --output "$out" >> "$RUN_ROOT/$job.log" 2>&1
            "$PYTHON" tools/audit_roi_recovery.py --candidates "$out/candidates.json" --annotations "$DATA_ROOT/fdi53/val.json" --output "$out/audit" >> "$RUN_ROOT/$job.log" 2>&1
        done
        "$PYTHON" tools/audit_roi_supervision.py --cache "$CACHE_ROOT/cache_7_val" --head "$RUN_ROOT/$job/last.pt" --device cuda:0 --output "$RUN_ROOT/${job}_matching.json" >> "$RUN_ROOT/$job.log" 2>&1
        printf '%s,SUCCESS\n' "$job" >> "$RUN_ROOT/status.csv"
    done
done
date -Iseconds > "$RUN_ROOT/completed.txt"
