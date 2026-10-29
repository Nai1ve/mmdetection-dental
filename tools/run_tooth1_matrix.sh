#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
BASE=/root/autodl-tmp
DATA=$BASE/datasets/private_only
RUN=${DENTAL_RUN_ROOT:-$BASE/experiments/tooth1_$(date +%Y%m%d_%H%M%S)}
MM=$BASE/envs/ultralytics/bin/python
MD=/root/miniconda3/envs/mmdet/bin/python
export DENTAL_DATA_ROOT="$DATA"
mkdir -p "$RUN"
printf 'model,start,end,status,work_dir\n' > "$RUN/status.csv"
git rev-parse HEAD > "$RUN/code_commit.txt"
config=configs/_teeth_/faster-rcnn_r50_fpn_24e_private_tooth1.py
failures=0
run_faster() {
    "$MD" tools/train.py "$config" --work-dir "$work" || return $?
    for split in val internal_test external_test; do
        mkdir -p "$work/eval/$split"
        DENTAL_TEST_SPLIT="$split" "$MD" tools/test.py "$config" "$work/epoch_24.pth" \
            --work-dir "$work/eval/$split" --out "$work/eval/$split/predictions.pkl" || return $?
        "$MD" tools/analysis_tools/evaluate_dental_candidate_file.py \
            --predictions "$work/eval/$split/predictions.pkl" --annotations "$DATA/fdi53/$split.json" \
            --output "$work/eval/$split/candidate_metrics.json" || return $?
    done
}
for model in faster_rcnn yolo11m rtdetr-l; do
    work="$BASE/work_dirs/$(basename "$RUN")_$model"
    start=$(date --iso-8601=seconds)
    printf '%s,%s,,RUNNING,%s\n' "$model" "$start" "$work" >> "$RUN/status.csv"
    echo "START $model $work"
    if [[ "$model" == faster_rcnn ]]; then
        run_faster > "$RUN/$model.log" 2>&1
        code=$?
    else
        converted="$BASE/datasets/private_only_yolo_tooth1_v1"
        if [[ ! -f "$converted/conversion.json" ]]; then
            "$MM" tools/prepare_modern_dental.py --root "$DATA" --output "$converted" --view tooth1 > "$RUN/conversion.log" 2>&1 || { echo CONVERSION_FAILED; exit 2; }
        fi
        "$MM" tools/run_modern_dental.py --model "$model" --data-root "$DATA" --converted "$converted" \
            --work-dir "$work" --view tooth1 --batch 4 --epochs 24 --imgsz 960 > "$RUN/$model.log" 2>&1
        code=$?
    fi
    sed -i '$d' "$RUN/status.csv"
    printf '%s,%s,%s,%s,%s\n' "$model" "$start" "$(date --iso-8601=seconds)" "$code" "$work" >> "$RUN/status.csv"
    echo "END $model exit=$code"
    [[ $code -eq 0 ]] || failures=$((failures+1))
done
exit "$((failures>0))"
