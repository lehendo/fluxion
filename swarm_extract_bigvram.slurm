#!/bin/bash
# One-off variant of swarm_extract.slurm for large-protein OOM retries.
# 2026-08-15: 1b25_D (619 residues -> Chai-1's 768-token crop bucket) OOM'd
# repeatedly (indices 24,25,26 -- all 3 replicas) on untyped --gres=gpu:1
# draws landing on 14.56 GiB cards (TeslaT4 nodes on secondary). The
# --low-memory flag (swarm_extract_lowmem.slurm) does NOT fix this -- it
# only reduces peak VRAM in the upstream feature-embedding/token-embedder
# stages, not the trunk.forward() attention op that's actually OOMing
# (confirmed by re-reading extract_latents.py: low_memory/return_on_cpu is
# never passed into the trunk call). The one successful low-memory run was
# luck of landing on a bigger card, not the fix working.
#
# Real fix: request a GPU type known to have enough VRAM (H100, 80 GiB) so
# the draw never lands on a card too small for this protein's crop size.
# secondary has substantial H100 capacity (6+ nodes x 8 GPUs, mostly
# "mixed" i.e. partially free as of 2026-08-15), so this should schedule
# quickly despite being a typed request -- unlike the earlier H100/H200
# experience, which was on IllinoisComputes-GPU, a much scarcer pool.
# Not merged into the main swarm_extract.slurm since only this protein's
# 3 tasks need it.
#SBATCH --job-name=flux_swarm_fixed_bigvram
#SBATCH --output=logs/swarm_fixed_%A_%a.out
#SBATCH --error=logs/swarm_fixed_%A_%a.err
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

mkdir -p logs

source activate fluxion
cd /u/arjunc4/fluxion

ATLAS_RAW=/scratch/arjunc4/atlas_raw
OUT_DIR=/scratch/arjunc4/fluxion_data/latents_fixed

# Build flat list of all (protein, replica) pairs -- identical to
# swarm_extract.slurm so array indices line up exactly.
PAIRS=()
for protein_dir in "$ATLAS_RAW"/*/; do
    protein_id=$(basename "$protein_dir")
    for r in 1 2 3; do
        xtc="$protein_dir/${protein_id}_prod_R${r}_fit.xtc"
        if [ -f "$xtc" ]; then
            PAIRS+=("${protein_id} ${r}")
        fi
    done
done

PAIR="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
PROTEIN_ID=$(echo "$PAIR" | awk '{print $1}')
REPLICA=$(echo "$PAIR"   | awk '{print $2}')

PDB="$ATLAS_RAW/$PROTEIN_ID/${PROTEIN_ID}.pdb"
XTC="$ATLAS_RAW/$PROTEIN_ID/${PROTEIN_ID}_prod_R${REPLICA}_fit.xtc"

echo "Worker $SLURM_ARRAY_TASK_ID: $PROTEIN_ID replica $REPLICA (H100, big-VRAM mode)"
echo "  PDB: $PDB"
echo "  XTC: $XTC"

python data_pipeline/extract_latents.py \
    --pdb          "$PDB"                     \
    --traj         "$XTC"                     \
    --out-dir      "$OUT_DIR/$PROTEIN_ID"     \
    --shard-prefix "${PROTEIN_ID}_R${REPLICA}" \
    --shard-size   500                        \
    --skip-pair
