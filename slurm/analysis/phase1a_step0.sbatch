#!/bin/bash
# Step 0: reproduce the old split's numbers under both token weightings. CPU only.
#SBATCH --job-name=p1a_step0
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/p1a_step0_%j.out
#SBATCH --error=logs/p1a_step0_%j.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
python scripts/phase1a_fit.py --step0 \
    --cache /scratch/arjunc4/fluxion_data/phase1_cache \
    --out-dir /scratch/arjunc4/fluxion_data/phase1_results \
    --device cpu
