#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs reports
TRAIN_PYTHON="$HOME/RAG/clm_training_env/bin/python"
INFER_PYTHON="${INFER_PYTHON:-$HOME/miniconda3/envs/rag-author/bin/python}"
test -x "$TRAIN_PYTHON"
test -x "$INFER_PYTHON"
echo 'START_MERGE: details in logs/merge.log'
"$TRAIN_PYTHON" -u merge.py > logs/merge.log 2>&1
echo 'START_SMOKE: details in logs/smoke.log'
CUDA_VISIBLE_DEVICES=0,1,2,3 "$INFER_PYTHON" -u run.py smoke > logs/smoke.log 2>&1
"$INFER_PYTHON" collect_reports.py
echo 'CLM_STEP6_SMOKE_COMPLETE rows=5'
