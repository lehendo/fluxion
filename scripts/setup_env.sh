#!/bin/bash
# scripts/setup_env.sh
#
# Full, tested recreation of the "fluxion" conda environment on the Illinois
# Campus Cluster. requirements.txt alone is NOT sufficient to rebuild this
# environment -- it can't express the Python version or the one non-pip
# (binary) dependency, aria2, which data_pipeline/download_atlas.py needs.
# This script is the single source of truth for all of it.
#
# Background: the fluxion env has been wiped multiple times by ICC's
# scratch-purge policy (conda environments must live in home, never
# scratch -- see ~/.conda's target). Each time, the rebuild recipe had to be
# rediscovered piece by piece. This script captures exactly what was
# verified working together on 2026-08-13 (Python 3.10.20, torch
# 2.6.0+cu124, chai-lab 0.6.1) so that never has to happen again.
#
# Usage:
#   bash scripts/setup_env.sh
#
# Safe to run on the login node -- everything here is network/IO-bound
# (package downloads), not CPU-bound, so it will not hit the login node's
# CPU-time limit. Run from anywhere; it cd's to the repo root itself.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== 1. Creating conda env 'fluxion' (Python 3.10) ==="
source ~/.bashrc
if conda env list | grep -qE '^fluxion\s'; then
    echo "  env 'fluxion' already exists -- skipping creation. Remove it first"
    echo "  (conda env remove -n fluxion) if you want a fully fresh rebuild."
else
    conda create -n fluxion python=3.10 -y
fi

conda activate fluxion

echo
echo "=== 2. Installing pinned Python dependencies from requirements.txt ==="
echo "    (torch/torchaudio/torchvision resolve from the CUDA 12.4 wheel index"
echo "    via the --extra-index-url directive in requirements.txt itself)"
pip install -r requirements.txt

echo
echo "=== 3. Installing aria2 (non-pip binary dependency, for download_atlas.py) ==="
conda install -c conda-forge aria2 -y

echo
echo "=== 4. Verifying the environment ==="
python -c "
import torch, chai_lab, MDAnalysis, h5py, pandas, wandb, filelock, torchdiffeq, scipy, tqdm, requests, matplotlib
from multiprocessing import resource_sharer
from torch.utils.data import DataLoader
print('torch', torch.__version__)
print('chai_lab', chai_lab.__version__ if hasattr(chai_lab, \"__version__\") else 'OK')
print('MDAnalysis', MDAnalysis.__version__)
print('ALL PYTHON IMPORTS PASSED')
"
which aria2c
aria2c --version | head -1

echo
echo "=== Setup complete. Environment 'fluxion' is ready. ==="
