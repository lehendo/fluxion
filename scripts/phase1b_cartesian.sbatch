#!/bin/bash
# B1 Cartesian control, one array task per protein (CPU). Array restricted to the 8 proteins that PASSED verification criterion (iii)
# (1ab1_A 1ah7_A 1aol_A 1b2s_F 1bq8_A 1btk_A 1bxy_A 1bxy_B); the other 11 are unverified (see EXPERIMENT_LOG.md, 2026-10-10).
#SBATCH --job-name=p1b_cart
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=2,4,6,10,14,15,17,18
#SBATCH --output=logs/p1b_cart_%A_%a.out
#SBATCH --error=logs/p1b_cart_%A_%a.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
PROTEINS=(16pk_A 1a62_A 1ab1_A 1af7_A 1ah7_A 1ail_A 1aol_A 1b0n_A 1b25_D 1b2s_E 1b2s_F 1b8d_K 1bgf_A 1bkp_A 1bq8_A 1btk_A 1bx7_A 1bxy_A 1bxy_B)
python scripts/phase1b_cartesian.py --protein "${PROTEINS[$SLURM_ARRAY_TASK_ID]}"
