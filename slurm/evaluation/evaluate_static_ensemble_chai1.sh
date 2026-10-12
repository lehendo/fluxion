#!/bin/bash
# Real 3D-structure evaluation of the static-ensemble pivot, using the
# validated Chai-1 diffusion decoder (see scripts/verify_chai1_decode.py,
# passed at RMSD=0.79A against ground truth 2026-07) instead of the cheaper
# nearest-neighbor decode. This is the actual test of generative novelty —
# NN-decode can only ever return a real training structure, so it can't
# tell us whether the model produces genuinely novel, physically valid
# conformations. Chai-1 decode can.
#
# Reuses the 1000-sample ensemble already generated for the NN-decode eval
# (results/static_ensemble_1bxy_A/1bxy_A_ensemble.pt) — no need to
# regenerate it, only the decode step changes.
#
# Needs a FULL GPU (~40GB), not a MIG slice — same requirement as the
# verification job. Real diffusion sampling per structure (~8-10s each,
# cached model loading across the loop — see inference/visualize.py's
# _CHAI1_COMPONENT_CACHE) means ~1000 x ~9s =~2.5h total; budgeted 4h.
#SBATCH --job-name=flux_eval_chai1
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_chai1_%j.out
#SBATCH --error=logs/eval_chai1_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

PROTEIN_ID="1bxy_A"

python scripts/evaluate_ensemble.py \
    --ensemble    results/static_ensemble_${PROTEIN_ID}/${PROTEIN_ID}_ensemble.pt \
    --latents-dir /scratch/arjunc4/fluxion_data/latents \
    --atlas-raw   /scratch/arjunc4/atlas_raw \
    --out-dir     results/static_ensemble_${PROTEIN_ID}_chai1 \
    --n-real-samples 1000 \
    --decode-mode chai1 \
    --num-diffn-timesteps 200 \
    --device      cuda
