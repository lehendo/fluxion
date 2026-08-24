#!/bin/bash
# Full-set pairwise (Mantel-style) validation sweep across all 19 pilot
# proteins, following the 2026-08-16 star-frame-0 sweep's mixed result.
# Spot-check on 7 proteins showed the star metric was largely an artifact
# (6/7 flipped from weak/negative to strong under pairwise) -- this
# confirms it holds across the full set before deciding on the
# latents_fixed/ -> latents/ swap. CPU-only, no GPU needed.
#SBATCH --job-name=flux_verify_pairwise_sweep
#SBATCH --output=logs/verify_pairwise_sweep_%j.out
#SBATCH --error=logs/verify_pairwise_sweep_%j.err
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

mkdir -p logs
source activate fluxion
cd /u/arjunc4/fluxion

LATENTS_DIR=/scratch/arjunc4/fluxion_data/latents_fixed
ATLAS_RAW=/scratch/arjunc4/atlas_raw

for protein_dir in "$ATLAS_RAW"/*/; do
    protein_id=$(basename "$protein_dir")
    echo "=================================================================="
    echo "PROTEIN: $protein_id"
    echo "=================================================================="
    python scripts/verify_latent_pairwise_sensitivity.py \
        --latents-dir "$LATENTS_DIR" \
        --atlas-raw   "$ATLAS_RAW"   \
        --protein-id  "$protein_id"  \
        --n-load      1500           \
        --n-sample    50
    echo
done
