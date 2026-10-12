#!/bin/bash
# Step 0 ladder (S0.0-S0.3). Needs phase1_cache_s0all (dense lag-1 cache, all positions) and phase1_results/acf.json.
#SBATCH --job-name=p1a_ladder
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/p1a_ladder_%j.out
#SBATCH --error=logs/p1a_ladder_%j.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
python scripts/phase1a_ladder.py
