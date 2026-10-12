# Fluxion: OT-CFM in Frozen Chai-1 Latent Space for Protein Conformational Ensembles

> **CORRECTION NOTICE (2026-10-10).** The "OT-CFM" label in this README describes `training/train_static.py` (static-ensemble pivot, minibatch OT coupling of noise to equilibrium frames) only.
> The temporal DiT trained by `training/train.py` (z0 = frame i, z1 = frame i+1) uses the *observed* adjacent-frame pairing with plain CFM regression and never calls
> `training/ot_coupling.py`. Results about the temporal model must not be attributed to OT coupling. The project's scientific
> position has also changed since this README was written (the work is being reframed as a measurement-practice study of latent-space protein dynamics), so the project-history sections below describe the earlier framing.


> Generating protein conformational ensembles by learning a minibatch-optimal-transport conditional flow matching (OT-CFM) velocity field in Chai-1's internal trunk latent space, trained on ATLAS MD trajectories, decoded back to real 3D structure via Chai-1's own diffusion module.

---

## Architecture

### The Oracle (Frozen Encoder)

We use **Chai-1** as a frozen structural encoder. Its core model is a `ModuleWrapper` over a `jit_module` (TorchScript), so standard `register_forward_hook` on internal layers is **not available**. The extraction pipeline (`data_pipeline/extract_latents.py`) drives Chai-1's own trunk pipeline directly (feature embedding → token input embedder → trunk recycles), pulling:

- **Single representation:** `[L, 384]` — per-residue token embeddings
- **Pair representation:** `[L, L, 256]` — residue–residue relationship tensor

These latents are the input/output space for the generative model. Chai-1 weights are never updated.

### The Generative Model (DiT + OT-CFM)

A **Diffusion Transformer (DiT)**, `FluxionDiT` (~40M params, `training/dit_model.py`), trained to learn a velocity field between noise and real conformations in Chai-1 latent space:

```
v_θ(z_t, pair, t) → predicted velocity, t ∈ [0, 1]
z_t = (1-t)·z0 + t·z1
```

**Current formulation — static conformational ensemble (as of 2026-07):** `z0` is fresh Gaussian noise, `z1` is a real equilibrium-ensemble conformation drawn i.i.d. from ATLAS (no frame-to-frame pairing). This reflects a pivot away from the original temporal (frame-to-frame) formulation — see "Project history" below for why.

**Training objective — genuine OT-CFM, not plain CFM.** Standard conditional flow matching pairs each `(z0, z1)` by arbitrary batch index. `training/ot_coupling.py` instead computes the minibatch-optimal pairing via the Hungarian algorithm (exact linear assignment, padding-aware) before computing the loss — this is what makes "OT-CFM" an accurate name for the training objective, not just an aspirational one. See that module's docstring for the full mechanism and the diagnostic finding that motivated it.

Loss is plain CFM regression (`training/flow_matching.py`) on top of whatever pairing it's handed — the loss function itself is agnostic to temporal vs. static framing.

### Sampling

Given trained `FluxionDiT`, draw an ensemble by integrating `dz/dt = v_θ(z, pair, t)` from `t=0` (fresh noise) to `t=1` (a generated conformation) — `inference/sample_ensemble.py`. Fixed-step (Euler/Heun/RK4) or adaptive (`torchdiffeq`) integrators are available (`inference/rollout.py`); RK4 vs. Euler integration were compared during diagnosis and found to give identical results (not a sampler-quality issue).

### Decoding latents to 3D structure

Two decoders, `inference/visualize.py`:

1. **Nearest-neighbor (`nn`):** for each generated latent, return the closest real training frame's coordinates. Cheap, GPU-free, but caps novelty at retrieval — cannot produce a structure the training set didn't already contain, and has a "ceiling effect" that can mask real generative-quality problems.
2. **Chai-1 diffusion module (`chai1`):** substitute the generated `(single, pair)` representation for Chai-1's own trunk output (monkeypatching only `load_exported("trunk.pt", ...)`), then run Chai-1's real, unmodified diffusion inference pipeline (`chai_lab.chai1.run_folding_on_context` — feature embedding, EDM/Karras multi-step denoising, confidence scoring, CIF writing) to produce genuine 3D coordinates. Validated against real ground-truth coordinates for a known ATLAS frame (`scripts/verify_chai1_decode.py`; RMSD ≈ 0.79 Å). Requires ~40 GB GPU memory; a real multi-step generative decode (not a single forward pass), so ~2–4 s/sample.

Evaluation against real MD data (`scripts/evaluate_ensemble.py`) computes ensemble diversity (pairwise RMSD distribution vs. real), RMSF profile correlation, and PCA projection overlap — using the Chai-1 decoder, not nearest-neighbor, since NN decode cannot detect mode-collapse or a diversity deficit (it can only ever return structures already in the training set).

---

## Project history: why "static ensemble," and why OT coupling

The project's original formulation was temporal: predict frame `i+1` from frame `i` (`training/train.py`). An exhaustive diagnostic chain (static/linear/velocity/nonlinear predictability probes, all at every lag tested from 10ps to 50ns) found **no exploitable frame-to-frame dynamics signal** in single/paired frozen Chai-1 snapshots — the real trained model never beat a trivial baseline. The project pivoted to generating **static i.i.d. conformational ensembles** instead (`training/train_static.py`), which reuses nearly all of the original infra (DiT, flow-matching loss, checkpointing, DDP/SLURM setup) and only changes what `z0` is and how the dataset is sampled.

Once evaluated with the real Chai-1 decoder (not nearest-neighbor, which masked the problem), the static-ensemble model showed a genuine diversity/mode-coverage deficit: uniform variance underestimation on one protein, generated samples landing outside the real ensemble's occupied conformational modes on another, larger one. Two sampling-time fixes (RK4 vs. Euler integration; noise-scale rescaling) were tested and ruled out. Root cause: the training objective, despite being named "OT-CFM" from early in the project, had never actually implemented minibatch optimal-transport coupling — plain index-order pairing produces an MSE-optimal target that is the conditional expectation over ambiguous, contradictory `(z0, z1)` targets, which compresses learned variance and can miss distinct conformational modes entirely. `training/ot_coupling.py` implements the real fix, and a retrain with it enabled is the actively evaluated model. See `docs/novelty_assessment.md` for how this affects the project's novelty claims and project memory for the full diagnostic chain.

`training/train.py` (temporal) is kept as a reference/ablation baseline, not because the temporal objective is expected to work.

---

## Infrastructure (UIUC Campus Cluster)

### Storage Layout

```
/u/arjunc4/fluxion/                       ← this repo (100 GB home quota)
/scratch/arjunc4/fluxion_data/
  ├── latents/<protein_id>/               ← per-protein extracted latents
  │     ├── manifest.json                 ← shard index (replica-tagged)
  │     ├── pair.pt                       ← static pair rep, float16
  │     ├── <protein>.fasta
  │     ├── esm_cache/
  │     └── <protein>_R<n>_shard_NNNNNN.h5
  ├── norm_stats.pt                       ← per-channel mean/std (compute_norm_stats.py)
  ├── checkpoints_static/                 ← original static-ensemble run (pre-OT-coupling)
  └── checkpoints_static_otcfm/           ← OT-coupling retrain (separate, not resumed from the above)
```

### HDF5 Sharding (Critical IO Constraint)

Raw latents are **never saved as individual `.pt` files**. The NFS metadata overhead would crash the cluster scheduler. Instead, `extract_latents.py` writes sharded `.h5` files grouped by up to `--shard-size` frames per shard (default 1,000), one shard file per `single` (and optionally `pair`) dataset, with a shared `manifest.json` per protein guarded by a filelock so concurrent per-replica SLURM jobs can append safely.

### SLURM-Aware Kill Switch (Training)

Training jobs are launched via `torchrun` under `srun --signal=SIGUSR1@120`. The training loop installs a `SIGUSR1` handler; on receipt, it checkpoints on rank 0 and exits cleanly before the walltime kill (`SIGKILL`) arrives. A `--max-steps` flag provides a separate, deliberate short-run exit for smoke tests/diagnostics.

### Checkpoint Resume with Exact RNG State

Checkpoints (`training/checkpoint.py`) save and restore:
- DiT model `state_dict`, EMA model `state_dict`
- Optimizer and LR scheduler `state_dict`
- `torch.get_rng_state()` / `torch.cuda.get_rng_state_all()`
- `numpy.random.get_state()` / `random.getstate()`
- Global step, epoch

Atomic write (`checkpoint.tmp` → `os.replace` → `checkpoint.pt`) so a mid-write kill never corrupts the last good checkpoint.

---

## Module Layout

```
fluxion/
├── data_pipeline/
│   ├── extract_latents.py     # Chai-1 trunk pipeline → HDF5 shards, per-protein/replica
│   ├── extract_pairs.py       # One static pair.pt per protein (frame 0, replica 1)
│   ├── dataset.py             # LatentDataset (frame pairs) + StaticFrameDataset (i.i.d. frames)
│   ├── compute_norm_stats.py  # Per-channel mean/std for latent normalization
│   ├── clean_latents.py       # Remove corrupted shards/manifests, keep pair.pt/fasta/esm_cache
│   └── download_atlas.py      # ATLAS trajectory download via DSIMB REST API
│
├── training/
│   ├── dit_model.py           # FluxionDiT — DiT architecture, pair-biased attention
│   ├── flow_matching.py       # CFM loss (agnostic to what z0/z1 represent)
│   ├── ot_coupling.py         # Minibatch OT coupling (Hungarian algorithm, padding-aware)
│   ├── checkpoint.py          # Atomic save/restore with full RNG state
│   ├── train.py                # ORIGINAL temporal (frame i -> frame i+1) training loop
│   └── train_static.py        # CURRENT static-ensemble (noise -> equilibrium sample) loop
│
├── inference/
│   ├── rollout.py              # ODE integrators (euler/heun/rk4/adaptive) + model loading
│   ├── sample_ensemble.py      # Draw a static ensemble from noise via the trained model
│   ├── visualize.py            # nn/chai1 decoders, Cα extraction, RMSD, plotting
│   └── infer.py                # End-to-end: checkpoint -> rollout -> decoded 3D trajectory
│
├── scripts/
│   ├── verify_chai1_decode.py       # Ground-truth validation of the Chai-1 decoder itself
│   ├── evaluate_ensemble.py         # Diversity / RMSF-correlation / PCA-overlap vs. real MD
│   ├── evaluate_checkpoint.py       # Trained-vs-trivial-baseline loss + gate wake-up check
│   ├── check_extraction.py          # Swarm extraction completeness report
│   ├── compute_trivial_baseline.py  # Zero-velocity baseline (temporal diagnostics)
│   ├── compute_position_mean_baseline.py  # Static per-position mean baseline (temporal diagnostics)
│   ├── compute_velocity_baseline.py       # Explicit finite-difference velocity baseline
│   ├── compute_restoring_force_baseline.py
│   ├── compute_nonlinear_probe.py   # MLP probe on (z0, z_prev) -> z1, held-out validation
│   └── summarize_lag_sweep.py       # Aggregate lag-sweep diagnostic logs into one table
│
├── tests/                      # pytest suite (unit + integration), run before any cluster job
├── requirements.txt
└── README.md
```

The `compute_*_baseline.py` / `compute_nonlinear_probe.py` scripts are diagnostic artifacts from the temporal-formulation investigation (see "Project history" above) — kept as the record of how the pivot decision was reached, not part of the active static-ensemble pipeline.

---

## Key Citations

- **Chai-1:** Chai Discovery, 2024
- **ATLAS:** Vander Meersche et al., 2024
- **Flow Matching:** Lipman et al., 2022; Albergo & Vanden-Eijnden, 2022
- **Minibatch OT-CFM:** Tong et al., 2023, "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport"
- **DiT:** Peebles & Xie, 2023

See `docs/novelty_assessment.md` for the full literature review establishing Fluxion's novelty against the closest competing methods (ConfRover, MDGen, BioEmu, Mac-Diff, and others).

---

## Status

- [x] Chai-1 latent extraction pipeline (`extract_latents.py`, `extract_pairs.py`) — all 19 ATLAS pilot proteins
- [x] `FluxionDiT` architecture, CFM loss, DDP/SLURM training infra, checkpointing
- [x] Temporal (frame-to-frame) formulation — exhaustively diagnosed, found to carry no exploitable signal at any tested lag; pivoted away from
- [x] Static-ensemble pivot (`train_static.py`, `StaticFrameDataset`) — trained and evaluated
- [x] Chai-1 real diffusion decoder (`chai1_decode_to_cif`) — implemented, validated against ground truth
- [x] Diversity/mode-coverage deficit diagnosed via real-decode evaluation; sampling-time fixes ruled out
- [x] Minibatch OT coupling implemented (`ot_coupling.py`) and unit-tested
- [ ] OT-coupling retrain — **in progress**; re-evaluation against the same Chai-1-decode pipeline is the deciding open question
- [ ] Full ATLAS scale-up, held-out-protein validation split, and paper design doc — held until the OT-coupling result is in
