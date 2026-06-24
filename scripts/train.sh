#!/bin/bash
#SBATCH --job-name=fluxion_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/arjunc4/fluxion_data/logs/train_%j.out
#SBATCH --error=/scratch/arjunc4/fluxion_data/logs/train_%j.err

# Run interactively: srun --pty bash, then manually execute this script.
# The Python training loop reads SLURM_JOB_START_TIME and saves a
# checkpoint + exits cleanly if within 15 min of walltime.

source ~/.bashrc
conda activate fluxion

cd /u/arjunc4/fluxion

python training/train.py \
    --scratch-dir /scratch/arjunc4/fluxion_data \
    --time-limit-hours 23.75 \
    "$@"
