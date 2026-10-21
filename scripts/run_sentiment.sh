#!/bin/bash
# Sentiment control, GPT-2 base, positive steering (+5) on the 2.5K negative prompts
# (README section 2.2 with --model gpt2). Usage: scripts/run_sentiment.sh smoke|full OUT_DIR [DEVICE_FLAG]
# smoke caps: 200 training steps, first 20 prompts (x 25 generations).
MODE=$1
OUT=$2
DEV=${3:---cuda}
mkdir -p $OUT
if [ "$MODE" = smoke ]; then CAPS_T="--n_steps 200 --log_step 10"; CAPS_G="--eval_size 20";
else CAPS_T="--n_steps 1000 --log_step 10"; CAPS_G=""; fi
export PYTHONUNBUFFERED=1
t0=$(date +%s); echo "TRAIN_START $(date -u +%FT%TZ)"
PYTHONPATH=. python experiments/training/train.py \
    --dataset_name sentiment-sst5 --ckpt_name $OUT/checkpoint.pt \
    --model gpt2 $DEV --adaptor_class multiply --num_steers 2 --dummy_steer 1 --rank 1000 \
    --batch_size 32 --max_length 256 --lr 1e-2 --regularization 1e-6 --epsilon 1e-3 --seed 0 \
    $CAPS_T > $OUT/train.log 2>&1
rc=$?; echo "TRAIN_EXIT $rc $(( $(date +%s)-t0 ))s $(date -u +%FT%TZ)"; [ $rc -ne 0 ] && exit $rc
t1=$(date +%s)
PYTHONPATH=. python experiments/training/generate.py \
    --eval_file data/prompts/sentiment_prompts-10k/negative_prompts.jsonl \
    --output_file $OUT/predictions-negative_5.jsonl --ckpt_name $OUT/checkpoint.pt \
    --model gpt2 $DEV --adaptor_class multiply --num_steers 2 --rank 1000 \
    --max_length 256 --verbose --steer_values 5 1 --top_p 0.9 \
    $CAPS_G > $OUT/generate.log 2>&1
rc=$?; echo "GEN_EXIT $rc $(( $(date +%s)-t1 ))s $(date -u +%FT%TZ)"; [ $rc -ne 0 ] && exit $rc
t2=$(date +%s)
python experiments/evaluation/evaluate.py \
    --generations_file $OUT/predictions-negative_5.jsonl \
    --metrics sentiment,ppl-big,dist-n \
    --output_file result_stats_negative_5.txt > $OUT/evaluate.log 2>&1
rc=$?; echo "EVAL_EXIT $rc $(( $(date +%s)-t2 ))s $(date -u +%FT%TZ)"
echo "ALL_DONE $(( $(date +%s)-t0 ))s"
exit $rc
