#!/bin/bash
# Generate + evaluate a static conformational ensemble from the CURRENT
# static-ensemble training checkpoint (checkpoints_static/checkpoint.pt,
# updated every 500 steps — safe to read mid-training since save_checkpoint
# writes atomically via os.replace()).
#
# One-shot, stateless job — no checkpoint of its own to protect against
# preemption, and trivially rerunnable in full, so scavenger is fine here
# despite the general "no scavenger for jobs that write data" rule (that
# rule targets long, expensive, hard-to-resume writes like shard extraction;
# this is a short inference + eval pass).
#
# NOTE: this training run used ALL 19 proteins with no held-out split, so
# this evaluates IN-DISTRIBUTION fit only — it can catch mode collapse and
# gross distributional mismatch, but does not prove generalization to
# unseen proteins. A proper held-out-protein eval requires a retrain with a
# train/val split.
#
# Adjust PROTEIN_ID and --gres below to whatever's actually free (check
# `sinfo -N -o "%25P %15N %25G %10t" | grep -i gpu`) if this partition/gres
# combo isn't available when you submit.
#SBATCH --job-name=flux_eval_ensemble
#SBATCH --partition=ic-express
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_ensemble_%j.out
#SBATCH --error=logs/eval_ensemble_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

PROTEIN_ID="1bxy_A"
CHECKPOINT=/scratch/arjunc4/fluxion_data/checkpoints_static/checkpoint.pt
LATENTS_DIR=/scratch/arjunc4/fluxion_data/latents
ATLAS_RAW=/scratch/arjunc4/atlas_raw
OUT_DIR=results/static_ensemble_${PROTEIN_ID}

mkdir -p "$OUT_DIR"

python inference/sample_ensemble.py \
    --checkpoint  "$CHECKPOINT" \
    --latents-dir "$LATENTS_DIR" \
    --protein-id  "$PROTEIN_ID" \
    --n-samples   1000 \
    --batch-size  32 \
    --n-steps     100 \
    --method      euler \
    --out         "$OUT_DIR/${PROTEIN_ID}_ensemble.pt" \
    --device      cuda

python scripts/evaluate_ensemble.py \
    --ensemble    "$OUT_DIR/${PROTEIN_ID}_ensemble.pt" \
    --latents-dir "$LATENTS_DIR" \
    --atlas-raw   "$ATLAS_RAW" \
    --out-dir     "$OUT_DIR" \
    --n-real-samples 1000
