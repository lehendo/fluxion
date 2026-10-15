#!/bin/bash
# (iii-L) frame-exact local alignment of the 10 ps coordinates with the latents; one array task per protein (CPU). Output: one JSON per protein.
#SBATCH --job-name=t2_localalign
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-18
#SBATCH --output=logs/t2_localalign_%A_%a.out
#SBATCH --error=logs/t2_localalign_%A_%a.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
PROTEINS=(16pk_A 1a62_A 1ab1_A 1af7_A 1ah7_A 1ail_A 1aol_A 1b0n_A 1b25_D 1b2s_E 1b2s_F 1b8d_K 1bgf_A 1bkp_A 1bq8_A 1btk_A 1bx7_A 1bxy_A 1bxy_B)
P=${PROTEINS[$SLURM_ARRAY_TASK_ID]}
python scripts/verify_atlas_local_alignment.py --proteins "$P" --out /scratch/arjunc4/fluxion_data/phase1_results/atlas_10ps_local_alignment/$P.json
