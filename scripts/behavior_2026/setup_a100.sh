#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
"$PYTHON_BIN" -m venv .venv-behavior
source .venv-behavior/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r scripts/behavior_2026/requirements-a100.txt
python -m pip install --no-deps -e '.[behavior]'
python -m pip check
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check driver/wheel"; assert torch.cuda.is_bf16_supported(), "BF16 GPU required"; print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)'
python -m unittest discover -s tests -p 'test_behavior*.py' -v
python -m pip freeze > .venv-behavior/installed-requirements.txt
