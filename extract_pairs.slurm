#!/bin/bash
#SBATCH --job-name=flux_pairs
#SBATCH --output=logs/pairs_%A_%a.out
#SBATCH --error=logs/pairs_%A_%a.err
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=48G
#SBATCH --array=0-18   # one job per protein (19 proteins)

mkdir -p logs

source activate fluxion
cd /u/arjunc4/fluxion

python data_pipeline/extract_pairs.py \
    --atlas-raw /scratch/arjunc4/atlas_raw \
    --out-dir   /scratch/arjunc4/fluxion_data/latents \
    --device    cuda
