#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv-behavior/bin/activate
: "${RUN_DIR:?Set a new output directory}"
if [[ -n "${RESUME_CKPT:-}" || -n "${INITIAL_WEIGHTS:-}" ]]; then
    echo 'Unset RESUME_CKPT and INITIAL_WEIGHTS for this joint-training comparison.' >&2
    exit 1
fi
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_PROJECT="${WANDB_PROJECT:-TechnyongTurbo}"
python -u -m turbovla.training.trainer \
    --split-manifest "${SPLIT_MANIFEST:-experiments/behavior_2026/toggle5_split.json}" \
    --task-balanced \
    --reference-config "${REFERENCE_CONFIG:-outputs/task0-batch32-epoch1/config.json}" \
    --dinov3 "${DINOV3_MODEL_PATH:-/workspace/models/dinov3-vitb}" \
    --bert "${BERT_MODEL_PATH:-/workspace/models/bert-base-uncased}" \
    --out "$RUN_DIR" --device cuda --vision-precision bf16_autocast \
    --image-size 224 --horizon 12 --lr 1e-4 \
    --batch-size "${BATCH_SIZE:-32}" --grad-accumulation "${GRAD_ACCUMULATION:-1}" \
    --workers "${NUM_WORKERS:-0}" --steps "${STEPS:-12048}" \
    --save-every "${SAVE_EVERY:-3012}" "$@"
