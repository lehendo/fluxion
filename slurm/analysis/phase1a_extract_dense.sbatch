#!/bin/bash
# Dense lag-1 cache for the Step 0 ladder: 3000 starts x 50 rows UNIFORM OVER ALL POSITIONS (incl. bucket
# padding, flagged is_pad) = 150k rows/protein, matching the old probe's pooling.
# CPU only (no DiT). One array task per protein.
#SBATCH --job-name=p1a_dense
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-18
#SBATCH --output=logs/p1a_dense_%A_%a.out
#SBATCH --error=logs/p1a_dense_%A_%a.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
ulimit -n 65536
PROTEINS=(16pk_A 1a62_A 1ab1_A 1af7_A 1ah7_A 1ail_A 1aol_A 1b0n_A 1b25_D 1b2s_E 1b2s_F 1b8d_K 1bgf_A 1bkp_A 1bq8_A 1btk_A 1bx7_A 1bxy_A 1bxy_B)
P=${PROTEINS[$SLURM_ARRAY_TASK_ID]}
python scripts/phase1a_extract.py \
    --latents-dir /scratch/arjunc4/fluxion_data/latents \
    --protein "$P" --out-dir /scratch/arjunc4/fluxion_data/phase1_cache_s0all \
    --lags 1 --frames-per-lag 3000 --rows-per-frame 50 --all-positions --num-workers 8 --device cpu
