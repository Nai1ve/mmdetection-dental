#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VARIANT="${1:-baseline}"
case "$VARIANT" in baseline|ccsw) ;; *) echo 'Use baseline or ccsw' >&2; exit 2;; esac
: "${DENTAL_DATA_ROOT:?Set DENTAL_DATA_ROOT to an extracted private dataset}"
export DENTAL_DATA_ROOT
WORK_DIR="${DENTAL_WORK_DIR:-/root/autodl-tmp/work_dirs/private_frcnn_$VARIANT}"
CONFIG="configs/_teeth_/faster-rcnn_r50_fpn_24e_private_${VARIANT}.py"
mkdir -p "$WORK_DIR"
python tools/train.py "$CONFIG" --work-dir "$WORK_DIR"
CHECKPOINT="$WORK_DIR/epoch_24.pth"
test -f "$CHECKPOINT"
if [ -f "$DENTAL_DATA_ROOT/fdi53/external_test.json" ]; then
    SPLITS=(val internal_test external_test)
else
    SPLITS=(val external_fuxingmen external_shijitan external_huairou)
fi
for SPLIT in "${SPLITS[@]}"; do
    mkdir -p "$WORK_DIR/eval/$SPLIT"
    DENTAL_TEST_SPLIT="$SPLIT" python tools/test.py "$CONFIG" "$CHECKPOINT" \
        --work-dir "$WORK_DIR/eval/$SPLIT" --out "$WORK_DIR/eval/$SPLIT/predictions.pkl"
done
