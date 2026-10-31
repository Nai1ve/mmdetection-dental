#!/usr/bin/env bash
set -euo pipefail
cd "${REPO:-/root/mmdetection-dental}"
BASE=${BASE:-/root/autodl-tmp}
PY=${PY:-$BASE/envs/ultralytics/bin/python}
RUN=${RUN:?Use a fresh output directory}
CHECKPOINT=${CHECKPOINT:?Use the original single-class YOLO checkpoint}
mkdir -p "$(dirname "$RUN")"
mkdir "$RUN"
echo $$ > "$RUN/runner.pid"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
git rev-parse HEAD > "$RUN/commit.txt"
sha256sum tools/dental_roi_head.py tools/run_dental_roi.py tools/run_roi_pilot.sh "$CHECKPOINT" > "$RUN/source_sha256.txt"
printf 'job,state\n' > "$RUN/status.csv"
trap 'printf "controller,FAILED\n" >> "$RUN/status.csv"' ERR
for split in train val; do
  printf 'cache_%s,RUNNING\n' "$split" >> "$RUN/status.csv"
  "$PY" tools/run_dental_roi.py cache --checkpoint "$CHECKPOINT" --split "$split" \
    --output "$RUN/cache_$split" > "$RUN/cache_$split.log" 2>&1
  printf 'cache_%s,SUCCESS\n' "$split" >> "$RUN/status.csv"
done
for mode in single multi; do
  printf '%s,RUNNING\n' "$mode" >> "$RUN/status.csv"
  "$PY" tools/run_dental_roi.py train --cache "$RUN/cache_train" --mode "$mode" \
    --output "$RUN/$mode" --epochs 6 --seed 42 > "$RUN/$mode.log" 2>&1
  for threshold in .25 .5; do
    "$PY" tools/run_dental_roi.py eval --cache "$RUN/cache_val" --head "$RUN/$mode/last.pt" \
      --threshold "$threshold" --output "$RUN/${mode}_val_$threshold" >> "$RUN/$mode.log" 2>&1
  done
  printf '%s,SUCCESS\n' "$mode" >> "$RUN/status.csv"
done
date --iso-8601=seconds > "$RUN/completed.txt"
