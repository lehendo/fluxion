#!/bin/bash
# Verify the fetched full-stride ATLAS trajectories against the extraction log and the 100 ps analysis archives (criteria (i)-(iii) pre-registered in the experiment log).
#SBATCH --job-name=t2_verify10ps
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/t2_verify10ps_%j.out
#SBATCH --error=logs/t2_verify10ps_%j.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
python scripts/verify_atlas_10ps.py
