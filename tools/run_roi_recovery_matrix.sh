#!/usr/bin/env bash
set -Eeuo pipefail
# Launch with setsid; never overwrite the run directory or a live controller.
: "${RUN_ROOT:?Set a fresh RUN_ROOT under /root/autodl-tmp/experiments}"
: "${DETECTOR:?Set frozen baseline checkpoint path}"
PYTHON=${PYTHON:-/root/autodl-tmp/envs/ultralytics/bin/python}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets/private_only}
EPOCHS=${EPOCHS:-6}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir "$RUN_ROOT"
echo $$ > "$RUN_ROOT/runner.pid"
git rev-parse HEAD > "$RUN_ROOT/commit.txt"
sha256sum "$DETECTOR" tools/dental_roi_head.py tools/run_dental_roi.py tools/audit_roi_recovery.py > "$RUN_ROOT/source_sha256.txt"
printf 'job,state\n' > "$RUN_ROOT/status.csv"
trap 'printf "controller,FAILED\n" >> "$RUN_ROOT/status.csv"' ERR
for size in 7 14; do
    for split in train val; do
        printf 'cache_%s_%s,RUNNING\n' "$size" "$split" >> "$RUN_ROOT/status.csv"
        "$PYTHON" tools/run_dental_roi.py cache --checkpoint "$DETECTOR" --data-root "$DATA_ROOT" --split "$split" --roi-size "$size" --output "$RUN_ROOT/cache_${size}_${split}" > "$RUN_ROOT/cache_${size}_${split}.log" 2>&1
        printf 'cache_%s_%s,SUCCESS\n' "$size" "$split" >> "$RUN_ROOT/status.csv"
    done
done
# Resolution, conditioning and quality are isolated before their combination.
for job in control7 resolution14 conditioned7 quality7 combined14; do
    size=7; flags=()
    case "$job" in
        resolution14) size=14 ;;
        conditioned7) flags=(--conditioned) ;;
        quality7) flags=(--quality) ;;
        combined14) size=14; flags=(--conditioned --quality) ;;
    esac
    printf '%s,RUNNING\n' "$job" >> "$RUN_ROOT/status.csv"
    "$PYTHON" tools/run_dental_roi.py train --cache "$RUN_ROOT/cache_${size}_train" --output "$RUN_ROOT/$job" --mode multi --epochs "$EPOCHS" --seed 42 "${flags[@]}" > "$RUN_ROOT/$job.log" 2>&1
    for threshold in .25 .5; do
        out="$RUN_ROOT/${job}_val_${threshold}"
        "$PYTHON" tools/run_dental_roi.py eval --cache "$RUN_ROOT/cache_${size}_val" --head "$RUN_ROOT/$job/last.pt" --data-root "$DATA_ROOT" --threshold "$threshold" --output "$out" >> "$RUN_ROOT/$job.log" 2>&1
        "$PYTHON" tools/audit_roi_recovery.py --candidates "$out/candidates.json" --annotations "$DATA_ROOT/fdi53/val.json" --output "$out/audit" >> "$RUN_ROOT/$job.log" 2>&1
    done
    printf '%s,SUCCESS\n' "$job" >> "$RUN_ROOT/status.csv"
done
date -Iseconds > "$RUN_ROOT/completed.txt"
# No automatic extra training, external-set selection or shutdown.
