# Fluxion: Latent Flow Matching for Millisecond Molecular Dynamics

> Predicting continuous protein dynamics and conformational breathing at millisecond timescales using conditional flow matching over foundation-model latent spaces.

---

## Architecture

### The Oracle (Frozen Encoder)

We use **Chai-1** as a frozen structural encoder. Its core model is a `ModuleWrapper` over a `jit_module` (TorchScript), so standard `register_forward_hook` on internal layers is **not available**. The extraction pipeline (`data_pipeline/extract_latents.py`) intercepts outputs via the official Chai-1 inference API, pulling:

- **Single representation:** `[L, d_single]` — per-residue token embeddings
- **Pair representation:** `[L, L, d_pair]` — residue–residue relationship tensor

These latents are the input/output space for the generative model. Chai-1 weights are never updated.

### The Generative Model (DiT + CFM)

A **Diffusion Transformer (DiT)** trained to learn the vector field of protein conformational change in latent space:

```
v_θ(z_t, t, dt) → Δz   such that   z(t+dt) = z(t) + Δz
```

The training objective is **Conditional Flow Matching (CFM)**: given consecutive frames `z(t)` and `z(t+dt)` from the ATLAS dataset, the model learns to regress the interpolated velocity field, enabling ODE-based rollout at inference time.

### Temporal Autoregression

- **Input:** current latent `z(t)`, time conditioning `t`, time delta `dt`
- **Output:** predicted next latent `z(t+dt)`
- **Rollout:** integrate with a neural ODE solver (`torchdiffeq`) for arbitrary-length trajectories

---

## Infrastructure

### Storage Layout (UIUC Campus Cluster)

```
/u/arjunc4/fluxion/          ← this repo (100 GB home quota)
/scratch/arjunc4/fluxion_data/
  ├── conda_env/             ← Conda environment (symlinked)
  ├── hf_cache/              ← HuggingFace model cache (symlinked)
  ├── raw_trajectories/      ← ATLAS .dcd + .pdb files
  ├── latents/               ← HDF5 shards of extracted latents
  └── checkpoints/           ← DiT training checkpoints
```

### HDF5 Sharding (Critical IO Constraint)

Raw latents are **never saved as individual `.pt` files**. The NFS metadata overhead would crash the cluster scheduler. Instead, `extract_latents.py` writes sharded `.h5` files grouped by ~1,000 frames per shard:

```
latents/protein_1ABC/shard_000000.h5
latents/protein_1ABC/shard_000001.h5
```

Each shard contains datasets: `single` `[N, L, d_single]`, `pair` `[N, L, L, d_pair]`, `timestamps` `[N]`.

### SLURM-Aware Kill Switch (Training)

Because training jobs are launched **manually inside `tmux` + `srun --pty bash`** sessions, the training loop reads:

```python
os.environ.get('SLURM_JOB_START_TIME')   # epoch seconds
os.environ.get('SLURM_JOB_END_TIME')     # epoch seconds (derived from walltime)
```

If the script detects it is within **15 minutes of walltime expiry**, it saves a full checkpoint and calls `sys.exit(0)`. A `--time-limit-hours` CLI flag provides a fallback for non-SLURM environments.

### Checkpoint Resume with Exact RNG State

Checkpoints save and restore:
- DiT model `state_dict`
- Optimizer `state_dict`
- LR scheduler `state_dict`
- `torch.get_rng_state()` / `torch.cuda.get_rng_state()`
- `numpy.random.get_state()`
- Global step, epoch, loss history

This guarantees that a resumed run sees exactly the same data order and augmentations as if it had never been interrupted.

---

## Module Layout

```
fluxion/
├── data_pipeline/
│   ├── extract_latents.py   # Chai-1 JIT-safe extraction → HDF5 shards
│   ├── dataset.py           # PyTorch Dataset over HDF5 shards
│   └── preprocess.py        # ATLAS .dcd → frame loader (MDAnalysis)
│
├── training/
│   ├── dit_model.py         # Diffusion Transformer architecture
│   ├── flow_matching.py     # CFM loss and interpolation utilities
│   ├── checkpoint.py        # Save/restore with full RNG state
│   └── train.py             # Main training loop + SLURM kill switch
│
├── inference/
│   ├── rollout.py           # ODE-based trajectory generation
│   └── visualize.py         # Trajectory visualization utilities
│
├── scripts/
│   ├── extract_latents.sh   # SLURM batch script for data mining
│   └── train.sh             # SLURM batch script for DiT training
│
├── requirements.txt
└── README.md
```

---

## Key Citations (Planned)

- **Chai-1:** Chai Discovery, 2024
- **ATLAS:** Vander Meersche et al., 2024 (millisecond MD trajectories)
- **Flow Matching:** Lipman et al., 2022; Albergo & Vanden-Eijnden, 2022
- **DiT:** Peebles & Xie, 2023

---

## Status

- [x] Architecture design
- [x] Project scaffolding
- [ ] `extract_latents.py` — Chai-1 JIT-safe latent extraction
- [ ] `dataset.py` — HDF5 shard loader
- [ ] `dit_model.py` — DiT architecture
- [ ] `flow_matching.py` — CFM loss
- [ ] `train.py` — training loop with SLURM kill switch
- [ ] `rollout.py` — inference ODE rollout
