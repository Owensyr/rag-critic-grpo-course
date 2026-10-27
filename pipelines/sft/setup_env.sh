#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p reports
python -c 'import torch; assert torch.__version__.startswith("2.8.0"), torch.__version__; print("Reusing:", torch.__version__, torch.__file__)'
ENV_DIR="$HOME/RAG/clm_training_env"
if [ ! -f "$ENV_DIR/bin/python" ]; then
  python -m venv --system-site-packages "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install --timeout 120 --retries 5 -c constraints.txt -r requirements-training.txt
"$ENV_DIR/bin/python" -m pip freeze --all > reports/training_environment_freeze.txt
"$ENV_DIR/bin/python" run.py environment
