#!/bin/bash
#SBATCH --job-name=flux_pairs
#SBATCH --output=logs/pairs_%j.out
#SBATCH --error=logs/pairs_%j.err
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=48G

mkdir -p logs

source activate fluxion
cd /u/arjunc4/fluxion

python data_pipeline/extract_pairs.py \
    --atlas-raw /scratch/arjunc4/atlas_raw \
    --out-dir   /scratch/arjunc4/fluxion_data/latents \
    --all \
    --device    cuda
