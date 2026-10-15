#!/bin/bash
# B1 at 10 ps stride, one array task per protein (CPU). A task whose protein did not pass the 10 ps verification prints SKIPPED and exits 0.
# Submit chained after the verification: sbatch --dependency=afterany:<verify job> slurm/analysis/phase1b_cartesian_10ps.sbatch
#SBATCH --job-name=p1b_10ps
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-18
#SBATCH --output=logs/p1b_10ps_%A_%a.out
#SBATCH --error=logs/p1b_10ps_%A_%a.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
PROTEINS=(16pk_A 1a62_A 1ab1_A 1af7_A 1ah7_A 1ail_A 1aol_A 1b0n_A 1b25_D 1b2s_E 1b2s_F 1b8d_K 1bgf_A 1bkp_A 1bq8_A 1btk_A 1bx7_A 1bxy_A 1bxy_B)
python scripts/phase1b_cartesian_10ps.py --protein "${PROTEINS[$SLURM_ARRAY_TASK_ID]}"
