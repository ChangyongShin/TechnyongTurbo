#!/usr/bin/env bash
# Resume the selected staged policy without changing the rolled-back project.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/workspace/TechnyongTurbo/.venv-behavior/bin/python}"
RUN_DIR="${RUN_DIR:-/input/Technyong_TurboVLA/task0-staged-100k-$(date +%Y%m%d-%H%M%S)}"
CHECKPOINT="/input/Technyong_TurboVLA/active_policy/step_5000.pt"
CHECKPOINT_SHA256="08fb14be979a66c637ec9911002c97851ade7caab571587f61c9cff9793475d5"
CODE_COMMIT=bbba897

test -x "$PYTHON_BIN"
test -f "$CHECKPOINT"
if [[ -e "$RUN_DIR/code" ]] || [[ -e "$RUN_DIR/checkpoints" ]]; then
    echo "Choose a fresh RUN_DIR; existing code/checkpoints are preserved: $RUN_DIR" >&2
    exit 1
fi
mkdir -p "$RUN_DIR/code"
git archive "$CODE_COMMIT" | tar -x -C "$RUN_DIR/code"
git show origin/main:scripts/behavior_2026/train_task0_long.py > "$RUN_DIR/code/scripts/behavior_2026/train_task0_long.py"
printf 'Historical training code: %s\nSelected source checkpoint: %s\n' "$CODE_COMMIT" "$CHECKPOINT"
export PYTHONPATH="$RUN_DIR/code${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$RUN_DIR/code"
exec "$PYTHON_BIN" -u scripts/behavior_2026/train_task0_long.py \
    --checkpoint "$CHECKPOINT" --expected-sha256 "$CHECKPOINT_SHA256" \
    --out "$RUN_DIR/checkpoints" --steps 100000 --save-every 5000
