# Four-GPU DDP

**Setup.** `torchrun --standalone --nproc_per_node=4 -m fluxion.training.train ...` with `DistributedSampler`, four DataLoader workers per rank and gradient all-reduce in `DDP`. The 40M-parameter run used four A100s for about 71 hours.

**Failures met, and what fixed each.**

- **"Too many open files" after tens of thousands of steps.** Sixteen DataLoader workers pass tensors through shared-memory file descriptors, and the soft limit was 1,024. Fixed by `torch.multiprocessing.set_sharing_strategy("file_system")` in `train.py`, and `ulimit -n 65536` in the job script as a margin.
- **"Duplicate GPU detected" from NCCL on MIG slices.** With several ranks on one MIG-sliced device, all ranks landed on the same physical slice. We did not find the root cause; the workaround is a single process per slice, or full GPUs for multi-rank runs.
- **Host memory grows faster than linearly with ranks.** Each rank runs its own workers and pinned buffers. One rank fitted in 96 GB, two needed 160 GB, and the four-rank run was given 384 GB. Extrapolate generously.
- **A node with no visible GPUs.** The job died within seconds with an internal assertion naming zero devices, although SLURM had allocated four. Exclude a node that fails this way instead of retrying on it.
- **Deadlock at shutdown.** Ranks that stop on different collectives hang until the NCCL watchdog fires. See [slurm_preemption.md](slurm_preemption.md), item 5.

Set `NCCL_ASYNC_ERROR_HANDLING=1` so that a failed collective aborts the job and does not hang.
