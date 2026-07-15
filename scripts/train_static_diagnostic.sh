#!/bin/bash
# Cheap pre-validation for the static-ensemble pivot (noise -> equilibrium
# frame OT-CFM, see training/train_static.py) BEFORE committing to a full
# multi-day multi-GPU run. Single GPU, capped at --max-steps, short walltime.
# Confirms: no crashes in the new dataset/collate/training-loop wiring, and
# that loss is finite and trending down — not a claim of final results.
#SBATCH --job-name=flux_train_static_diag
#SBATCH --partition=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:QuadroRTX6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --signal=SIGUSR1@120
#SBATCH --output=logs/train_static_diag_%j.out
#SBATCH --error=logs/train_static_diag_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Same 14-protein pilot subset used for the temporal-formulation diagnostics,
# for direct comparability.
PILOT_PROTEINS="1a62_A,1ab1_A,1ah7_A,1ail_A,1b0n_A,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B"

torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    training/train_static.py \
    --scratch-dir   /scratch/arjunc4/fluxion_data \
    --proteins      "$PILOT_PROTEINS" \
    --batch-size    16 \
    --lr            1e-4 \
    --epochs        1000 \
    --log-interval  10 \
    --max-steps     3000
