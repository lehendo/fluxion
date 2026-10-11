#!/bin/bash
# Phase 3a evaluations at 25/50/75/100% of the run for both arms (protocol pre-registered in experiments/EXPERIMENT_LOG.md).
#SBATCH --job-name=p3_eval
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/p3_eval_%j.out
#SBATCH --error=logs/p3_eval_%j.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
P3=/scratch/arjunc4/fluxion_data/phase3
mkdir -p $P3/eval
for m in 100 025 050 075; do
    python scripts/phase3_eval.py --arm uniform $P3/uniform/milestone_$m.pt --arm beta $P3/beta/milestone_$m.pt --out $P3/eval/eval_$m.json
done
