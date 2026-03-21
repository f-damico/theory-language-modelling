#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-environment_theory_language_modelling_gpu_cu113.yml}"
ENV_NAME="theory-lm"

if [ ! -f "$ENV_FILE" ]; then
  echo "[ERROR] Missing env file: $ENV_FILE"
  exit 1
fi

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] Could not find conda.sh under ~/miniconda3 or ~/anaconda3"
  exit 1
fi

conda deactivate >/dev/null 2>&1 || true
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
conda env create -f "$ENV_FILE"

conda activate "$ENV_NAME"

python -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
python -m pip install --no-cache-dir --upgrade pip setuptools wheel
python -m pip install --no-cache-dir \
  torch==1.12.1+cu113 \
  torchvision==0.13.1+cu113 \
  torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu113

python - <<'PY'
import torch
print('torch_version =', torch.__version__)
print('cuda_runtime =', torch.version.cuda)
print('cuda_available =', torch.cuda.is_available())
print('cuda_device_count =', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu_name =', torch.cuda.get_device_name(0))
else:
    raise SystemExit('ERROR: torch installed, but CUDA is still unavailable')
PY
