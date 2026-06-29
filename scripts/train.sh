#!/bin/bash
#SBATCH --job-name=flux_train
#SBATCH --partition=ic-express
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100.1g.20gb:4
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
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=29500

# Pilot: 14 fast proteins (~2 sec/frame on H100). Excludes 1b25_D, 16pk_A, 1af7_A, 1bkp_A, 1aol_A.
PILOT_PROTEINS="1a62_A,1ab1_A,1ah7_A,1ail_A,1b0n_A,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B"

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    training/train.py \
    --scratch-dir /scratch/arjunc4/fluxion_data \
    --proteins     "$PILOT_PROTEINS" \
    --batch-size 4 \
    --lr 1e-4 \
    --epochs 200 \
    --log-interval 10
