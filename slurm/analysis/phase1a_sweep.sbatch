#!/bin/bash
# Lag sweep (10 ps - 50 ns; 50 ns reported greyed/not interpretable), Phase 1C, asymptote check. GPU for the MLPs.
#SBATCH --job-name=p1a_sweep
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/p1a_sweep_%j.out
#SBATCH --error=logs/p1a_sweep_%j.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
python scripts/phase1a_fit.py
