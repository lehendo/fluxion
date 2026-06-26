#!/bin/bash
#SBATCH --job-name=flux_train
#SBATCH --partition=ic-express            # Migrated to the express queue for Hopper GPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100.1g.20gb:4         # Scaling to 4 H100/H200 GPUs for DDP
#SBATCH --cpus-per-task=16                # Allocating CPU threads to handle h5py workers
#SBATCH --mem=128G                        # Ensuring ample system RAM for file handle caching
#SBATCH --time=04:00:00                   # Maximum time limit for the partition
#SBATCH --requeue                         # Automatically place job back in queue if preempted
#SBATCH --signal=SIGUSR1@120              # Send SIGUSR1 to Python exactly 120 seconds before death
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Ensure logs directory exists
mkdir -p logs

# Activate cluster environment
source ~/.bashrc
conda activate fluxion

# Move into repository root
cd /u/arjunc4/fluxion

# --- Performance & Interconnect Optimization Variables ---
export OMP_NUM_THREADS=4                  # Matches cpus-per-task per process to prevent CPU thrashing
export NCCL_ASYNC_ERROR_HANDLING=1       # Prevents hangs if a single GPU drops during preemption
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=29500

echo "Starting distributed training across 4 Hopper GPUs..."
echo "Master node running on: $MASTER_ADDR"

# --- Launch DDP via torchrun ---
srun torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    training/train.py \
    --scratch-dir /scratch/arjunc4/fluxion_data \
    --batch-size 8 \
    --lr 1e-4 \
    --epochs 200 \
    --log-interval 10
