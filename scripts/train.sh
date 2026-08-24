#!/bin/bash
#SBATCH --job-name=flux_train
#SBATCH --partition=ic-express
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.20gb:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1

# 2026-08-24: diagnostic-scale re-run on corrected latents/ (post
# coordinate-injection fix + validated swap). Expanded from the old
# 14-protein wall-clock-limited subset to the full 19-protein set (all
# now have corrected latents). Judge this run's loss against today's
# diagnostic-chain baselines on the same normalized latent space:
# trivial (v_pred=0) floor = 0.004596, linear (OLS on z0,z_prev) held-out
# baseline = 0.001638 -- the DiT needs to meaningfully beat the linear
# number to prove its extra capacity (pair conditioning, nonlinearity) is
# adding value, not just the trivial floor.
PILOT_PROTEINS="16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B"

srun torchrun \
    --standalone \
    --nproc_per_node=4 \
    training/train.py \
    --scratch-dir /scratch/arjunc4/fluxion_data \
    --proteins     "$PILOT_PROTEINS" \
    --batch-size 4 \
    --lr 1e-4 \
    --epochs 200 \
    --log-interval 10
