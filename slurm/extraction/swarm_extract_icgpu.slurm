#!/bin/bash
# One-off variant of swarm_extract.slurm for 1b25_D (indices 24,25,26),
# the large-protein OOM/slow-scheduling problem child.
# 2026-08-16: swarm_extract_bigvram.slurm (secondary, typed/constrained
# to large-VRAM GPUs) worked correctly but needed repeated resubmission
# every 4h (secondary's hard wall-time cap) across 5+ rounds for this one
# protein. Switching to the IllinoisComputes-GPU partition instead:
#   - MaxTime=3-00:00:00 (72h) per `scontrol show partition
#     IllinoisComputes-GPU` -- one submission should cover this protein's
#     full run instead of restarting from scratch each 4h window.
#   - Its node pool (ccc0387-0390, ccc0465) is EXCLUSIVELY A100 (x4/node)
#     and H200 (x8/node) -- no small-VRAM cards exist on this partition
#     at all, so untyped --gres=gpu:1 is safe here without needing the
#     feature-constraint OR-list swarm_extract_bigvram.slurm needed on
#     secondary's mixed pool. (Feature tags are inconsistent across
#     partitions anyway -- this A100 node doesn't even carry a standalone
#     "A100" feature, only a combined string -- so constraint-matching
#     would've been fragile here regardless.)
#   - Confirmed access: arjunc4 is in group acc-IllinoisComputes (one of
#     the partition's AllowGroups), and AllowAccounts=ALL/AllowQos=ALL,
#     so the existing jimeng-ic account works unchanged.
#   - ccc0390 shows State=drained (unusable) as of 2026-08-16; the other
#     4 nodes are live, with A100 nodes ccc0387/0389 in "mixed" (real
#     idle capacity) at submission time.
# Not merged into swarm_extract_bigvram.slurm or the main
# swarm_extract.slurm since only this protein's 3 tasks need it.
#SBATCH --job-name=flux_swarm_fixed_icgpu
#SBATCH --output=logs/swarm_fixed_%A_%a.out
#SBATCH --error=logs/swarm_fixed_%A_%a.err
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
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

echo "Worker $SLURM_ARRAY_TASK_ID: $PROTEIN_ID replica $REPLICA (IllinoisComputes-GPU, 72h)"
echo "  PDB: $PDB"
echo "  XTC: $XTC"

python data_pipeline/extract_latents.py \
    --pdb          "$PDB"                     \
    --traj         "$XTC"                     \
    --out-dir      "$OUT_DIR/$PROTEIN_ID"     \
    --shard-prefix "${PROTEIN_ID}_R${REPLICA}" \
    --shard-size   500                        \
    --skip-pair
