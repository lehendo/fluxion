#!/bin/bash
# Diagnostic: is the low diversity_ratio (0.36) from the Chai-1-decoded
# ensemble eval (2026-07, job 9701978) a SAMPLING-time artifact (crude
# first-order Euler ODE integration under-dispersing samples — a
# well-documented flow-matching/diffusion sampler bias) or a genuine
# TRAINING-time limitation?
#
# Re-samples the SAME trained checkpoint with RK4 (4th-order accurate,
# already implemented in inference/rollout.py, unused until now) instead of
# Euler, then re-runs the identical Chai-1-decode evaluation. No retraining.
#
# If diversity improves substantially -> sampler was the bottleneck, fix is
# trivial. If it stays low -> genuine training-time limitation, worth
# deeper investigation. Either way this is decisive, cheap information.
#SBATCH --job-name=flux_resample_rk4
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/resample_rk4_%j.out
#SBATCH --error=logs/resample_rk4_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

PROTEIN_ID="1bxy_A"
CHECKPOINT=/scratch/arjunc4/fluxion_data/checkpoints_static/checkpoint.pt
LATENTS_DIR=/scratch/arjunc4/fluxion_data/latents
ATLAS_RAW=/scratch/arjunc4/atlas_raw
OUT_DIR=results/static_ensemble_${PROTEIN_ID}_rk4

mkdir -p "$OUT_DIR"

echo "=== Stage 1: resample with RK4 (vs. the original Euler run) ==="
python inference/sample_ensemble.py \
    --checkpoint  "$CHECKPOINT" \
    --latents-dir "$LATENTS_DIR" \
    --protein-id  "$PROTEIN_ID" \
    --n-samples   1000 \
    --batch-size  32 \
    --n-steps     100 \
    --method      rk4 \
    --out         "$OUT_DIR/${PROTEIN_ID}_ensemble.pt" \
    --device      cuda

echo "=== Stage 2: evaluate via the validated Chai-1 diffusion decoder ==="
python scripts/evaluate_ensemble.py \
    --ensemble    "$OUT_DIR/${PROTEIN_ID}_ensemble.pt" \
    --latents-dir "$LATENTS_DIR" \
    --atlas-raw   "$ATLAS_RAW" \
    --out-dir     "${OUT_DIR}_chai1" \
    --n-real-samples 1000 \
    --decode-mode chai1 \
    --num-diffn-timesteps 200 \
    --device      cuda
