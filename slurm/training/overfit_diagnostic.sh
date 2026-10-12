#!/bin/bash
# Classic "can it overfit a tiny fixed batch" sanity check -- see
# scripts/overfit_diagnostic.py's docstring for full reasoning. Cheap:
# one fixed batch, no dataloader-worker overhead, should finish in
# minutes even with a generous step count.
#
# 2026-08-27: moved from secondary (Priority-queued, no ETA) to
# ic-express, which had a confirmed-idle MIG slice at submit time
# (sinfo showed ccc0482 idle). Safe here unlike the earlier full
# training runs: this script is a plain single-process script (no
# torchrun/DDP, num_workers=0), so it never hits the MIG + multi-
# process "Duplicate GPU detected" bug that forced those runs down to
# --nproc_per_node=1 in the first place -- that bug was specifically
# about multiple DDP ranks on one slice, not single-process jobs.
#SBATCH --job-name=flux_overfit_diag
#SBATCH --partition=ic-express
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/overfit_diag_%j.out
#SBATCH --error=logs/overfit_diag_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

python scripts/overfit_diagnostic.py \
    --scratch-dir /scratch/arjunc4/fluxion_data \
    --protein 1a62_A \
    --batch-size 8 \
    --n-steps 1000 \
    --lr 1e-4 \
    --log-interval 25 \
    --device cuda
