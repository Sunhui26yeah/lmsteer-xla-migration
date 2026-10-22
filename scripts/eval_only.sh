#!/bin/bash
# Evaluation stage only, on generations saved by an earlier full run (resume).
# Usage: scripts/eval_only.sh SRC_RUN_DIR OUT_DIR [DEVICE_FLAG]
SRC=$1
OUT=$2
DEV=${3:---cuda}
EVAL_DEV=""; [ "$DEV" = "--xla" ] && EVAL_DEV="--device xla"
mkdir -p $OUT
cp $SRC/predictions-negative_5.jsonl $SRC/train.log $SRC/generate.log $OUT/
export PYTHONUNBUFFERED=1
t2=$(date +%s)
python experiments/evaluation/evaluate.py \
    --generations_file $OUT/predictions-negative_5.jsonl \
    --metrics sentiment,ppl-big,dist-n \
    --output_file result_stats_negative_5.txt $EVAL_DEV > $OUT/evaluate.log 2>&1
rc=$?; echo "EVAL_EXIT $rc $(( $(date +%s)-t2 ))s $(date -u +%FT%TZ)"
exit $rc
