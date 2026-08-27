#!/bin/bash
# One-off diagnostic, now serving two purposes:
#
# 1. (2026-08-25) Does a larger batch size change the loss trajectory over
#    the same number of steps, compared to 10151511 (the batch-4 full run)?
#    First attempt (10153220) got to step ~2160 (well past warmup) before
#    an OOM cut it short. Post-warmup mean loss (~0.00387) was statistically
#    indistinguishable from the batch-4 run's post-warmup mean (~0.00399-
#    0.00407) -- no evidence larger batch alone fixes the flat-loss plateau,
#    though the sample was small.
#
# 2. (2026-08-26 to 2026-08-27) Test a fix for the OOM itself. 10153220's
#    crash was a Linux cgroup oom_kill (host RAM past --mem, not CUDA OOM)
#    around step ~2000+, not immediately. First guess: glibc allocator
#    fragmentation from the 4 DataLoader workers handling variable-sized
#    pair tensors (L_max varies 256/384/512/768 per batch depending on
#    which proteins get sampled) -- tried MALLOC_ARENA_MAX=1 (job
#    10189207), which had ZERO effect: crashed at the IDENTICAL step
#    (2160) again. That result rules out glibc's allocator (which
#    MALLOC_ARENA_MAX controls) and points instead at PyTorch's own
#    pinned-memory allocator (pin_memory=True in train.py's DataLoader),
#    which is well-documented to cache a block per distinct tensor SIZE
#    and never release it -- unaffected by MALLOC_ARENA_MAX entirely.
#    Real fix (now in training/train.py + data_pipeline/dataset.py's
#    collate_fn): pad every batch's pair tensor to a FIXED length (768,
#    the largest Chai-1 crop bucket) instead of the per-batch dynamic
#    max, so the pinned-memory allocator only ever sees one size and
#    reuses the same cached block indefinitely. --max-steps stays at
#    5000 (past the ~2160 failure point) to confirm the real fix holds.
#
# --scratch-dir points at an ISOLATED directory (fluxion_data_diag/,
# latents/ symlinked to the real corrected data, no duplication) rather
# than the live /scratch/arjunc4/fluxion_data/ -- keeps this fully
# independent of any other training run's checkpoint state.
#SBATCH --job-name=flux_train_bs8diag
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_bs8diag_%j.out
#SBATCH --error=logs/train_bs8diag_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1

PILOT_PROTEINS="16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B"

srun torchrun \
    --standalone \
    --nproc_per_node=1 \
    training/train.py \
    --scratch-dir /scratch/arjunc4/fluxion_data_diag \
    --proteins     "$PILOT_PROTEINS" \
    --batch-size 8 \
    --lr 1e-4 \
    --epochs 1 \
    --log-interval 10 \
    --max-steps 5000
