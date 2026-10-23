# Checkpoint-on-preemption under SLURM

**Failure mode.** A training job that is preempted or reaches its wall limit loses every step since the last checkpoint. One run lost four hours of progress this way even though the program had a SIGUSR1 handler: the signal never reached it.

**What the fix needs, in order.** Each item below was a separate failure found in turn.

1. **Deliver the signal to the batch script, not to `srun`.** A bare `#SBATCH --signal=SIGUSR1@120` is sent to whatever `srun` tracks, here the `torchrun` launcher, and the default action of SIGUSR1 killed it before any rank's handler ran. Use `--signal=B:SIGUSR1@120`, which delivers to the batch script, and trap it there.
2. **Do not guess the process tree.** Matching the training command with `pkill -f` also hit every DataLoader worker (they inherit the handler through fork and can die mid-read). Signalling the direct children of `torchrun` hit a supervisor that is not a rank. The process tree under `torchrun` is not stable across versions.
3. **Let each rank report itself.** `fluxion.training.train` writes `$FLUXION_DATA/.rank_pids/$SLURM_JOB_ID/rank_<N>.pid` at start-up, and the trap in `examples/slurm_template.sbatch` forwards SIGUSR1 to exactly those PIDs.
4. **Keep the handler async-signal-safe.** The handler only sets a flag and prints. Calling `dist.get_rank()` inside it can deadlock silently if it interrupts NCCL internals holding a lock, so the rank is cached at start-up.
5. **Decide to stop collectively.** Signals arrive per rank at slightly different times. A rank that sees the flag first can leave the loop while another is still inside the step's gradient all-reduce, and the job hangs until NCCL's watchdog fires after 600 s. The loop therefore all-reduces the flag (MAX) at the top of each step, where all ranks are already synchronized, and only then saves and exits.
6. **Keep the batch script alive.** `wait $PID` returns immediately when a trapped signal interrupts it. The script then exits, SLURM tears down the job step, and the checkpoint is killed mid-write. The template re-waits in a loop until the child really exits.

**Also needed.** `#SBATCH --requeue` to restart the job, and checkpoints that contain model, EMA, optimizer, scheduler and all RNG states and are written atomically (`fluxion.training.checkpoint`), so that a resumed run continues the same trajectory.

**Where.** `src/fluxion/training/train.py` (handler, PID files, collective decision), `src/fluxion/training/checkpoint.py`, `examples/slurm_template.sbatch`.
