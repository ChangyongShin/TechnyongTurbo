#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${BEHAVIOR_CACHE:?Set the prepared cache root}"
: "${DINOV3_MODEL_PATH:?Set the local DINOv3 ViT-B model directory}"
: "${BERT_MODEL_PATH:?Set the local BERT directory}"
: "${TRAIN_EPISODES:?Set space-separated training episode IDs}"
: "${VAL_EPISODES:?Set disjoint validation episode IDs}"
: "${RUN_DIR:?Set a new output directory}"
source .venv-behavior/bin/activate
train_paths=()
validation_paths=()
read -r -a train_ids <<< "$TRAIN_EPISODES"
read -r -a val_ids <<< "$VAL_EPISODES"
for episode in "${train_ids[@]}"; do
    [[ "$episode" =~ ^[0-9]+$ ]] || { echo 'Invalid training episode ID' >&2; exit 1; }
    printf -v name 'episode_%06d' "$((10#$episode))"
    train_paths+=("$BEHAVIOR_CACHE/$name")
done
for episode in "${val_ids[@]}"; do
    [[ "$episode" =~ ^[0-9]+$ ]] || { echo 'Invalid validation episode ID' >&2; exit 1; }
    printf -v name 'episode_%06d' "$((10#$episode))"
    validation_paths+=("$BEHAVIOR_CACHE/$name")
done
extra=()
if [[ "${RGBD:-0}" == 1 ]]; then extra+=(--rgbd); fi
if [[ -n "${RESUME_CKPT:-}" ]]; then extra+=(--resume "$RESUME_CKPT"); fi
if [[ -n "${INITIAL_WEIGHTS:-}" ]]; then extra+=(--initial-weights "$INITIAL_WEIGHTS"); fi
python -m turbovla.training.trainer \
    --train "${train_paths[@]}" --validation "${validation_paths[@]}" \
    --dinov3 "$DINOV3_MODEL_PATH" --bert "$BERT_MODEL_PATH" --out "$RUN_DIR" \
    --device cuda --vision-precision bf16_autocast --image-size 224 --horizon 12 \
    --batch-size "${BATCH_SIZE:-2}" --grad-accumulation "${GRAD_ACCUMULATION:-4}" \
    --workers "${NUM_WORKERS:-0}" --steps "${STEPS:-1000}" --save-every "${SAVE_EVERY:-500}" \
    "${extra[@]}" "$@"
