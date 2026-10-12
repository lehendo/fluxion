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
#
# 2026-07 resubmission: the previous attempt (job 9724697) successfully
# decoded 295/300 samples via Chai-1 (the expensive, ~3h part) but crashed
# in the final aggregation step on 5 samples with a minority atom count —
# now fixed (evaluate_ensemble.py resumes from existing CIFs and filters
# minority-shape outliers instead of crashing). Stage 1 (ensemble
# generation) is INTENTIONALLY SKIPPED here: it has no fixed seed, so
# rerunning it would silently desync the ensemble from the already-decoded
# CIFs (which are matched to it by index, not content). Reuses the
# already-generated results/static_ensemble_1b25_D/1b25_D_ensemble.pt as-is.
#SBATCH --job-name=flux_eval_1b25D_resume
#SBATCH --partition=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:QuadroRTX6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/eval_1b25D_resume_%j.out
#SBATCH --error=logs/eval_1b25D_resume_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

PROTEIN_ID="1b25_D"
LATENTS_DIR=/scratch/arjunc4/fluxion_data/latents
ATLAS_RAW=/scratch/arjunc4/atlas_raw
OUT_DIR=results/static_ensemble_${PROTEIN_ID}

echo "=== Stage 2: evaluate via the validated Chai-1 diffusion decoder (resume) ==="
python scripts/evaluate_ensemble.py \
    --ensemble    "$OUT_DIR/${PROTEIN_ID}_ensemble.pt" \
    --latents-dir "$LATENTS_DIR" \
    --atlas-raw   "$ATLAS_RAW" \
    --out-dir     "${OUT_DIR}_chai1" \
    --n-real-samples 300 \
    --decode-mode chai1 \
    --num-diffn-timesteps 200 \
    --device      cuda
