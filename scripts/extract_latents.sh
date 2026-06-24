#!/bin/bash
#SBATCH --job-name=fluxion_extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/arjunc4/fluxion_data/logs/extract_%j.out
#SBATCH --error=/scratch/arjunc4/fluxion_data/logs/extract_%j.err

# Usage: sbatch extract_latents.sh --protein-id 1ABC --traj-dir <path>

source ~/.bashrc
conda activate fluxion

cd /u/arjunc4/fluxion

python data_pipeline/extract_latents.py \
    --scratch-dir /scratch/arjunc4/fluxion_data \
    --shard-size 1000 \
    "$@"
