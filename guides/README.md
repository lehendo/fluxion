# Infrastructure guides

Short notes on the reusable pieces of this codebase, each paired with the failure it fixes. They record what went wrong, not a design.

| guide | failure mode |
|---|---|
| [slurm_preemption.md](slurm_preemption.md) | a job killed at its wall limit loses all training progress |
| [ddp_four_gpu.md](ddp_four_gpu.md) | four-GPU DDP runs fail from descriptors, memory, MIG slices and drained nodes |
| [chai1_conditioning.md](chai1_conditioning.md) | per-frame coordinates silently never reached the frozen trunk |
| [frame_variance_gate.md](frame_variance_gate.md) | a degenerate dataset is extracted after hours of GPU time |
| [resumable_evaluator.md](resumable_evaluator.md) | an evaluation killed at the wall limit restarts from zero |
