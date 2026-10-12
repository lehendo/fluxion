#!/bin/bash
# One-off validation sweep: verify_latent_frame_sensitivity.py across all 19
# pilot proteins on the corrected latents_fixed/ extraction, before deciding
# whether to swap it in for latents/. CPU-only (no GPU needed -- the script
# reads .h5 shards directly), so no --gres line.
#SBATCH --job-name=flux_verify_sweep
#SBATCH --output=logs/verify_sweep_%j.out
#SBATCH --error=logs/verify_sweep_%j.err
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --time=02:00:00
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
    python scripts/verify_latent_frame_sensitivity.py \
        --latents-dir "$LATENTS_DIR" \
        --atlas-raw   "$ATLAS_RAW"   \
        --protein-id  "$protein_id"  \
        --n-frames    1500
    echo
done
