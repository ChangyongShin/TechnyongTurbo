#!/usr/bin/env bash
# Prepare the original full task0 split, then start a separate staged experiment.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/workspace/TechnyongTurbo/.venv-behavior/bin/python}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/behavior-2026-rgb}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/data/task0-audit-cache}"
BASELINE_CONFIG="${BASELINE_CONFIG:-/workspace/TechnyongTurbo/outputs/task0-batch32-epoch1/config.json}"
STEPS="${STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
RUN_DIR="${RUN_DIR:-/workspace/TechnyongTurbo/outputs/task0-staged-full-b64-5000}"
WARM_START_ARGS=()
DEFAULT_DECISION_PRETRAIN=100
if [[ -n "${WARM_START:-}" ]]; then
    test -f "$WARM_START"
    WARM_START_ARGS=(--warm-start "$WARM_START")
    DEFAULT_DECISION_PRETRAIN=0
fi

test -x "$PYTHON_BIN"
test -f "$BASELINE_CONFIG"
if [[ -e "$RUN_DIR" ]] && [[ -n "$(ls -A -- "$RUN_DIR")" ]]; then
    echo "Output already exists; choose a fresh RUN_DIR: $RUN_DIR" >&2
    exit 1
fi
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -u scripts/behavior_2026/prepare_task0_full.py \
    --root "$DATA_ROOT" --cache "$CACHE_ROOT" --baseline-config "$BASELINE_CONFIG"

# No --smoke: all 180 TRAIN and 20 held-out validation episodes are used.
"$PYTHON_BIN" -u scripts/behavior_2026/train_staged.py \
    --cache "$CACHE_ROOT" --baseline-config "$BASELINE_CONFIG" --out "$RUN_DIR" \
    --steps "$STEPS" --save-every "$SAVE_EVERY" --batch-size "$BATCH_SIZE" \
    --decision-pretrain-steps "${DECISION_PRETRAIN_STEPS:-$DEFAULT_DECISION_PRETRAIN}" \
    --lr "${LEARNING_RATE:-1e-4}" \
    --startup-seconds "${STARTUP_SECONDS:-10}" \
    --startup-fraction "${STARTUP_FRACTION:-0}" \
    --startup-base-weight "${STARTUP_BASE_WEIGHT:-0}" \
    "${WARM_START_ARGS[@]}" \
    --dinov3 "${DINOV3_MODEL_PATH:-/workspace/models/dinov3-vitb}" \
    --bert "${BERT_MODEL_PATH:-/workspace/models/bert-base-uncased}"
