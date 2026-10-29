#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/root/autodl-tmp/envs/ultralytics/bin/python
DATA=/root/autodl-tmp/datasets/private_only
CONVERTED=/root/autodl-tmp/datasets/private_only_yolo53_v1
RUN=/root/autodl-tmp/experiments/modern_detectors_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
printf 'model,start,end,exit_code\n' > "$RUN/status.csv"
if [[ ! -f "$CONVERTED/conversion.json" ]]; then
    "$PY" tools/prepare_modern_dental.py --root "$DATA" --output "$CONVERTED" > "$RUN/conversion.log" 2>&1 || exit 2
fi
failures=0
for model in yolo11m rtdetr-l; do
    start=$(date --iso-8601=seconds)
    work=/root/autodl-tmp/work_dirs/modern_53_${model}_$(date +%Y%m%d_%H%M%S)
    echo "START $model work=$work"
    "$PY" tools/run_modern_dental.py --model "$model" --data-root "$DATA" \
        --converted "$CONVERTED" --work-dir "$work" --batch 4 --epochs 24 --imgsz 960 > "$RUN/$model.log" 2>&1
    code=$?
    printf '%s,%s,%s,%s\n' "$model" "$start" "$(date --iso-8601=seconds)" "$code" >> "$RUN/status.csv"
    echo "END $model exit=$code"
    [[ $code -eq 0 ]] || failures=$((failures+1))
done
exit "$((failures>0))"
