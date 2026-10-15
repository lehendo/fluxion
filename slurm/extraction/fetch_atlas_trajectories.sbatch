#!/bin/bash
# Resumable download + integrity check of the full-stride (10 ps) ATLAS trajectory archives for the 19 pilot proteins (~7.3 GB, CPU only). Safe to resubmit: finished proteins are skipped, partial files resume.
#SBATCH --job-name=t2_atlas10ps
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/t2_atlas10ps_%j.out
#SBATCH --error=logs/t2_atlas10ps_%j.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
python scripts/fetch_atlas_trajectories.py
