#!/bin/bash
# Diagnostic: RK4 vs Euler showed IDENTICAL results (diversity_ratio 0.365
# vs 0.364), ruling out ODE integration precision as the cause of the
# under-diverse generated ensemble. Visual inspection (RMSF profile, PCA
# projection) shows a specific pattern: strong shape correlation (0.88) but
# UNIFORMLY compressed magnitude — classic variance underestimation in a
# conditionally-trained generative model, not broken/off-manifold
# generation.
#
# This tests whether that's recoverable at sampling time (no retraining):
# rescale the input noise z0 by a factor before ODE integration. Scale
# chosen from the observed diversity ratio: 1/sqrt(0.365) ~ 1.65 — if the
# learned noise->structure map is a roughly smooth, variance-compressing
# transform, this should directly compensate.
#
# If diversity recovers toward ~1.0 without degrading rmsf_correlation ->
# cheap fix, done. If not -> compression is a fixed property of the trained
# map; retraining-side intervention is genuinely justified next.
#SBATCH --job-name=flux_noise_scale
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/noise_scale_%j.out
#SBATCH --error=logs/noise_scale_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

PROTEIN_ID="1bxy_A"
CHECKPOINT=/scratch/arjunc4/fluxion_data/checkpoints_static/checkpoint.pt
LATENTS_DIR=/scratch/arjunc4/fluxion_data/latents
ATLAS_RAW=/scratch/arjunc4/atlas_raw
OUT_DIR=results/static_ensemble_${PROTEIN_ID}_noisescale165

mkdir -p "$OUT_DIR"

echo "=== Stage 1: resample with noise_scale=1.65 (Euler, same as original baseline) ==="
python inference/sample_ensemble.py \
    --checkpoint  "$CHECKPOINT" \
    --latents-dir "$LATENTS_DIR" \
    --protein-id  "$PROTEIN_ID" \
    --n-samples   1000 \
    --batch-size  32 \
    --n-steps     100 \
    --method      euler \
    --noise-scale 1.65 \
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
