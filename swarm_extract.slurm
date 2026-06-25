#!/bin/bash
#SBATCH --job-name=flux_swarm
#SBATCH --output=logs/swarm_%A_%a.out
#SBATCH --error=logs/swarm_%A_%a.err
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=48G
#SBATCH --array=0-3

# Ensure logs directory exists
mkdir -p logs

# Activate your environment
source activate fluxion

# Run the extraction
python data_pipeline/extract_latents.py \
    --pdb /scratch/arjunc4/raw_data/sample/1AKE.pdb \
    --dcd /scratch/arjunc4/raw_data/sample/1AKE.pdb \
    --out-dir /scratch/arjunc4/fluxion_data/latents \
    --shard-size 2
