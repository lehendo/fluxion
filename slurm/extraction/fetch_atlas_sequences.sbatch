#!/bin/bash
#SBATCH --job-name=p1_fetch_atlas
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/p1_fetch_atlas_%j.out
#SBATCH --error=logs/p1_fetch_atlas_%j.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
python scripts/fetch_atlas_sequences.py
