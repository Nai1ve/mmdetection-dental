#!/usr/bin/env bash
set -uo pipefail
cd /root/mmdetection-dental
BASE=/root/autodl-tmp
RUN=$BASE/experiments/tooth1_20261029_221247
RECOVERY=$RUN/evaluation_recovery
PY=$BASE/envs/ultralytics/bin/python
mkdir -p "$RECOVERY"
echo $$ > "$RECOVERY/runner.pid"
printf 'model,end,status\n' > "$RECOVERY/status.csv"
while ps -p 6628 -o args= | grep -q '^bash tools/run_tooth1_matrix.sh'; do sleep 20; done
failures=0
for model in yolo11m rtdetr-l; do
    work=$BASE/work_dirs/tooth1_20261029_221247_$model
    if ! "$PY" -c 'import csv,sys; rows=list(csv.DictReader(open(sys.argv[1]))); assert len(rows)==24 and float(rows[-1]["epoch"])==24' "$work/results.csv"; then
        printf '%s,%s,INCOMPLETE_TRAINING\n' "$model" "$(date --iso-8601=seconds)" >> "$RECOVERY/status.csv"
        failures=$((failures+1))
        continue
    fi
    "$PY" tools/run_modern_dental_eval_v2.py --model "$model" --data-root "$BASE/datasets/private_only" \
        --converted "$BASE/datasets/private_only_yolo_tooth1_v1" --work-dir "$work" \
        --view tooth1 --batch 4 --imgsz 960 --evaluate-only > "$RECOVERY/$model.log" 2>&1
    code=$?
    printf '%s,%s,%s\n' "$model" "$(date --iso-8601=seconds)" "$code" >> "$RECOVERY/status.csv"
    [[ $code -eq 0 ]] || failures=$((failures+1))
done
exit "$((failures>0))"
