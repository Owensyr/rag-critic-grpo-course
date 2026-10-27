#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
ENV_ROOT="${ENV_ROOT:-$HOME/.venvs}"
mkdir -p "$ENV_ROOT"
for name in rag-inference clm-training; do
  "$PYTHON_BIN" -m venv "$ENV_ROOT/$name"
  "$ENV_ROOT/$name/bin/python" -m pip install --upgrade pip
  "$ENV_ROOT/$name/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
done
"$ENV_ROOT/rag-inference/bin/python" -m pip install -c requirements/constraints.txt -r requirements/inference.txt
"$ENV_ROOT/clm-training/bin/python" -m pip install -c requirements/constraints.txt -r requirements/training.txt
"$ENV_ROOT/rag-inference/bin/python" -m pip check
"$ENV_ROOT/clm-training/bin/python" -m pip check
echo "ENVIRONMENTS_INSTALLED: configure configs/paths.json; GPU checks run in the training/inference workers."
