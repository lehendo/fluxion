#!/bin/bash
#SBATCH --job-name=flux_train
#SBATCH --partition=ic-express
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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

# Single GPU / single process: 10127034 hit an NCCL "Duplicate GPU
# detected" error across all 4 local ranks on this MIG-sliced node (all
# landed on the same physical device 4e000 despite torch.cuda.set_device
# per local_rank in train.py) -- a MIG + multi-process DDP interaction
# bug, not a data or model issue. Since this run's only purpose is
# checking the loss trajectory against today's diagnostic baselines
# (not throughput), dropping to 1 GPU/1 process sidesteps the whole bug
# class entirely -- no cross-process device binding to get wrong.
# 10141835 OOM-killed a DataLoader worker at --batch-size 16: pair
# tensors are padded to [B, L_max, L_max, 256] float32 per batch, and
# with 1b25_D (L=768) now in the pilot for the first time (the old
# 14-protein subset excluded it specifically), a single batch of 16 can
# hit ~9.6 GB, and train.py hardcodes num_workers=4 (prefetch_factor=2
# default) -- worst case ~9 batches in flight at once, ~87 GB for pair
# tensors alone, well past the 64G that was requested. Reverted
# --batch-size to 4 (the original, tested-safe per-GPU value from the
# 4-GPU config -- effective batch size no longer matches that config,
# but this run only needs to show whether the loss trends down and
# where it lands vs. today's baselines, not reproduce a specific
# hyperparameter regime) and bumped --mem to 96G for margin.
srun torchrun \
    --standalone \
    --nproc_per_node=1 \
    training/train.py \
    --scratch-dir /scratch/arjunc4/fluxion_data \
    --proteins     "$PILOT_PROTEINS" \
    --batch-size 4 \
    --lr 1e-4 \
    --epochs 200 \
    --log-interval 10
