#!/bin/bash
# Generalization check: does the diversity-compression signature found on
# 1bxy_A (60 residues, diversity_ratio=0.36, rmsf_correlation=0.88,
# unaffected by ODE integrator quality or noise-scale rescaling) also show
# up on a structurally different protein, or is it specific to 1bxy_A?
#
# 1b25_D chosen for maximum contrast: largest crop bucket (768 vs 1bxy_A's
# 256), likely a substantially larger/different protein.
#
# Same baseline settings as the original 1bxy_A run (Euler, noise_scale=1.0,
# n_steps=100) for direct comparability, EXCEPT n_samples reduced to 300:
# measured real decode rate on this larger protein (768 vs 256 tokens) is
# ~40.5s/sample (~4x slower than 1bxy_A), so 1000 samples would need ~11+
# hours and blow the original 4h budget with no partial output saved
# (evaluate_ensemble.py only writes results at the end of the loop) —
# resubmitted 2026-07 after cancelling the first, doomed-to-timeout attempt
# (job 9715471) before it wasted more compute. 300 samples is still
# statistically decisive for a yes/no generalization check — the 1bxy_A
# signature wasn't borderline.
#SBATCH --job-name=flux_eval_1b25D
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_1b25D_%j.out
#SBATCH --error=logs/eval_1b25D_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

PROTEIN_ID="1b25_D"
CHECKPOINT=/scratch/arjunc4/fluxion_data/checkpoints_static/checkpoint.pt
LATENTS_DIR=/scratch/arjunc4/fluxion_data/latents
ATLAS_RAW=/scratch/arjunc4/atlas_raw
OUT_DIR=results/static_ensemble_${PROTEIN_ID}

mkdir -p "$OUT_DIR"

echo "=== Stage 1: generate ensemble (Euler, noise_scale=1.0, same as original baseline) ==="
python inference/sample_ensemble.py \
    --checkpoint  "$CHECKPOINT" \
    --latents-dir "$LATENTS_DIR" \
    --protein-id  "$PROTEIN_ID" \
    --n-samples   300 \
    --batch-size  32 \
    --n-steps     100 \
    --method      euler \
    --out         "$OUT_DIR/${PROTEIN_ID}_ensemble.pt" \
    --device      cuda

echo "=== Stage 2: evaluate via the validated Chai-1 diffusion decoder ==="
python scripts/evaluate_ensemble.py \
    --ensemble    "$OUT_DIR/${PROTEIN_ID}_ensemble.pt" \
    --latents-dir "$LATENTS_DIR" \
    --atlas-raw   "$ATLAS_RAW" \
    --out-dir     "${OUT_DIR}_chai1" \
    --n-real-samples 300 \
    --decode-mode chai1 \
    --num-diffn-timesteps 200 \
    --device      cuda
