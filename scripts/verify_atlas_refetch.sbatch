#!/bin/bash
#SBATCH --job-name=p1b_verify
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/p1b_verify_%j.out
#SBATCH --error=logs/p1b_verify_%j.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
python scripts/verify_atlas_refetch.py
