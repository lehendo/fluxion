#!/bin/bash
# T3a (uncentered, padding-unmasked) pre-registered evaluation: one array task per milestone (0-3 -> 025 050 075 100). One-step only (--steps 1), no flow curves, no baselines
# (they live in centered coordinates and are read from the centered evaluation of the same frames). Scores REAL tokens only; PRIMARY metric is in-sample, held-out is secondary.
#SBATCH --job-name=p3_t3a_eval
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/p3_t3a_eval_%A_%a.out
#SBATCH --error=logs/p3_t3a_eval_%A_%a.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn)
MS=(025 050 075 100)
M=${MS[$SLURM_ARRAY_TASK_ID]}
P3=/scratch/arjunc4/fluxion_data/phase3
mkdir -p $P3/eval_sec
python scripts/phase3_eval.py --arm t3a $P3/t3a/milestone_$M.pt --out $P3/eval_sec/eval_t3a_$M.json --no-baselines --skip-flow --steps 1
