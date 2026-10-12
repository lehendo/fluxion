#!/bin/bash
# Validate the corrected Chai-1 diffusion decoder against real ground-truth
# coordinates (see scripts/verify_chai1_decode.py). This is a one-off
# correctness check, not a training/production job.
#
# Needs a FULL GPU, not a MIG slice — Chai-1's full pipeline (feature
# embedder + trunk substitute + diffusion module + confidence head loaded
# together) needs ~40GB, well beyond a 20GB MIG fraction.
#SBATCH --job-name=flux_verify_chai1
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verify_chai1_%j.out
#SBATCH --error=logs/verify_chai1_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

python scripts/verify_chai1_decode.py \
    --latents-dir /scratch/arjunc4/fluxion_data/latents \
    --atlas-raw   /scratch/arjunc4/atlas_raw \
    --protein-id  1bxy_A \
    --out-dir     results/verify_chai1_decode \
    --num-diffn-timesteps 200 \
    --pass-threshold 5.0 \
    --device      cuda
