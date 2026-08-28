#!/bin/bash
# Classic "can it overfit a tiny fixed batch" sanity check -- see
# scripts/overfit_diagnostic.py's docstring for full reasoning. Cheap:
# one fixed batch, no dataloader-worker overhead, should finish in
# minutes even with a generous step count.
#SBATCH --job-name=flux_overfit_diag
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
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
