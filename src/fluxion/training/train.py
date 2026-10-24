"""fluxion/training/train.py

DDP training loop for the temporal formulation: conditional flow matching from frame i to frame i+1 (z0 = frame i, z1 = frame i+1), with EMA, gate-learning-rate optimizer groups
and SIGUSR1 checkpointing. It is the formulation of the original run reported in the paper (Section 7.1), which recovered nothing; the controlled arms use training/train_t3.py.

Launch with torchrun (see examples/slurm_template.sbatch and guides/ddp_four_gpu.md):
    torchrun --nproc_per_node=NUM_GPUS -m fluxion.training.train --scratch-dir $FLUXION_DATA --proteins 16pk_A,1a62_A

SLURM preemption: send SIGUSR1 to every rank; the loop checkpoints cleanly before the wall-time kill arrives (guides/slurm_preemption.md).
"""

import os
import sys
import random
import signal
import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# 2026-09-04 fix: job 10279682 (the real 72h run) died after ~46,650 steps
# / ~1h with "OSError: [Errno 24] Too many open files". PyTorch's default
# multiprocessing sharing strategy ('file_descriptor') opens a new fd for
# every tensor handed from a DataLoader worker to the main process; with
# 4 ranks x num_workers=4 = 16 workers running for tens of thousands of
# steps, this eventually exhausts the process's open-file ulimit even
# though nothing in the training code itself leaked anything. This is a
# well-documented PyTorch/Linux gotcha (not specific to this codebase) --
# switching to 'file_system' avoids the fd-per-tensor handoff entirely.
# Paired with a ulimit bump in the sbatch wrapper scripts as a safety
# margin on top of this, per PyTorch's own recommended combination.
torch.multiprocessing.set_sharing_strategy("file_system")

from fluxion.model.dit import FluxionDiT
from fluxion.training.flow_matching import FlowMatcher
from fluxion.training.checkpoint import save_checkpoint, load_checkpoint
from fluxion.data_pipeline.dataset import build_multi_protein_dataset, collate_fn

# ---------------------------------------------------------------------------
# Global preemption flag — set by SIGUSR1 handler, checked each batch. Also
# cache this rank's ID at startup (see main(), section 1) rather than
# calling dist.get_rank() from inside the handler: a signal handler runs
# asynchronously and can interrupt the main thread mid-call into NCCL/
# torch.distributed internals, which are not reentrant/async-signal-safe --
# if the interrupted code held an internal lock that dist.get_rank() also
# needs, the handler deadlocks silently (no traceback, no print, just a
# process that stops responding). Confirmed as the likely cause of job
# 10214022: rank 1's handler printed fine, rank 0's never printed at all,
# and the job died with zero evidence of a crash -- consistent with one
# rank happening to get unlucky on lock timing and the other not.
# ---------------------------------------------------------------------------
PREEMPTION_REQUESTED = False
_RANK_FOR_SIGNAL_HANDLER = 0


def handle_sigusr1(signum, frame) -> None:
    global PREEMPTION_REQUESTED
    PREEMPTION_REQUESTED = True
    print(
        f"[Rank {_RANK_FOR_SIGNAL_HANDLER}] SLURM preemption signal received. "
        "Preparing to checkpoint and exit...",
        flush=True,
    )


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    """In-place EMA update: ema ← decay·ema + (1-decay)·model."""
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)


def get_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    lr: float,
    gate_lr_multiplier: float = 1.0,
) -> list[dict]:
    """
    Three groups:
      - "decay":    regular >=2D weights (attention/FFN/projections), with
                    weight decay, at the base learning rate.
      - "no_decay": remaining 1-D params (biases), no weight decay, at the
                    base learning rate.
      - "gate":     every zero-initialized "wake-up" parameter — the AdaLN-
                    zero modulation weight+bias in every block and in
                    final_layer, plus final_layer's own output-projection
                    weight+bias — no weight decay, at
                    gate_lr_multiplier * base learning rate.

    Why the gate group needs its own (higher) LR
    -----------------------------------------------
    Every block computes x = x + gate * f(x), where f is attention or FFN.
    The gradient reaching f's OWN internal parameters is scaled by gate's
    CURRENT value (chain rule through the multiplication) — while gate ~ 0,
    those parameters get essentially no usable signal. final_layer.linear
    plays the same throttling role for the ENTIRE stack: its still-small
    weight multiplies the gradient flowing backward into every earlier
    block, including their gates. But the gate/final-projection parameters
    themselves receive gradient through a DIFFERENT, undiminished path (as
    an outer multiplier on f(x)/norm(x), not gated by their own value) — so
    they can, and empirically do, learn on their own regardless of how
    starved the rest of the network is. Giving them a higher LR accelerates
    them past this cold-start bottleneck without over-accelerating the rest
    of the network's still mostly noise-dominated gradient.
    """
    decay, no_decay, gate = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "adaLN_mod" in name or "final_layer.linear" in name:
            gate.append(param)
        elif param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay,    "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0,           "lr": lr},
        {"params": gate,     "weight_decay": 0.0,           "lr": lr * gate_lr_multiplier},
    ]


def compute_group_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    """
    Must be called after loss.backward() and before optimizer.zero_grad().

    Splits the L2 gradient norm into two groups:
      "total_grad_norm": every parameter's gradient — this is the same
          quantity torch.nn.utils.clip_grad_norm_ measures pre-clip.
      "gate_grad_norm": only the AdaLN-zero gate/final-projection parameters
          (those that start at exactly zero — see get_param_groups for why).

    If gate_grad_norm is a substantial fraction of (or roughly equal to)
    total_grad_norm, that means the OPPOSITE of what "starved gates" would
    look like: virtually all gradient signal is concentrated in the gates,
    and the rest of the network (attention/FFN internals) is what's
    starved — because their gradient is chain-rule-multiplied by the still-
    tiny gate value that sits between them and the loss. This was confirmed
    empirically on this model (2026-07): gate_grad_norm tracked
    total_grad_norm almost exactly at every logged step.
    """
    total_sq = 0.0
    gate_sq = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        sq = param.grad.detach().double().pow(2).sum().item()
        total_sq += sq
        if "adaLN_mod" in name or "final_layer.linear" in name:
            gate_sq += sq
    return {
        "total_grad_norm": total_sq ** 0.5,
        "gate_grad_norm": gate_sq ** 0.5,
    }


def warmup_lambda(step: int, warmup_steps: int) -> float:
    """Linear warmup, then constant LR."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return 1.0


def main(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # 1. Distributed setup
    # ------------------------------------------------------------------
    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # ------------------------------------------------------------------
    # 2. SLURM signal handler
    # ------------------------------------------------------------------
    global _RANK_FOR_SIGNAL_HANDLER
    _RANK_FOR_SIGNAL_HANDLER = global_rank
    signal.signal(signal.SIGUSR1, handle_sigusr1)

    # Self-report this rank's PID so the SLURM wrapper script's SIGUSR1
    # forwarding can target it directly. torchrun's internal process tree
    # (launcher -> possible supervisor -> rank workers) isn't a stable,
    # externally-guessable shape across torchrun/PyTorch versions --
    # confirmed the hard way: `pkill -SIGUSR1 -P <torchrun_pid>` (targeting
    # torchrun's direct children) hit a single intermediate process that
    # was neither rank, killing the job before either rank's handler ever
    # ran. Self-reporting sidesteps process-tree discovery entirely.
    job_id = os.environ.get("SLURM_JOB_ID", "nojobid")
    pid_dir = Path(args.scratch_dir) / ".rank_pids" / job_id
    pid_dir.mkdir(parents=True, exist_ok=True)
    (pid_dir / f"rank_{global_rank}.pid").write_text(str(os.getpid()))

    # ------------------------------------------------------------------
    # 3. Model + EMA + optimizer + warmup scheduler
    # ------------------------------------------------------------------
    model     = FluxionDiT().to(device)
    ema_model = deepcopy(model).requires_grad_(False)

    optimizer = torch.optim.AdamW(
        get_param_groups(model, args.weight_decay, args.lr, args.gate_lr_multiplier),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_lambda(step, args.warmup_steps),
    )

    matcher = FlowMatcher()

    # ------------------------------------------------------------------
    # 4. Resume from checkpoint (before wrapping in DDP)
    # ------------------------------------------------------------------
    checkpoint_dir = Path(args.scratch_dir) / "checkpoints"
    step, start_epoch = load_checkpoint(
        checkpoint_dir / "checkpoint.pt",
        model,
        ema_model,
        optimizer,
        scheduler,
        device,
    )

    model = DDP(model, device_ids=[local_rank])

    # ------------------------------------------------------------------
    # 5. Dataset + DataLoader
    # ------------------------------------------------------------------
    latents_dir = Path(args.scratch_dir) / "latents"

    # Normalization stats are required, not optional. FlowMatcher's MSE loss
    # regresses directly against these latents; left unnormalized, a handful
    # of high-variance channels dominate the loss and starve gradient signal
    # for everything else — see data_pipeline/compute_norm_stats.py docstring.
    norm_stats_path = latents_dir / "norm_stats.pt"
    if not norm_stats_path.exists():
        proteins_flag = f' --proteins "{args.proteins}"' if args.proteins else ""
        raise FileNotFoundError(
            f"Normalization stats not found at {norm_stats_path}. Run:\n"
            f"  python data_pipeline/compute_norm_stats.py "
            f"--latents-dir {latents_dir}{proteins_flag}\n"
            f"before training."
        )

    protein_ids = [p.strip() for p in args.proteins.split(",")] if args.proteins else None
    dataset = build_multi_protein_dataset(latents_dir, protein_ids=protein_ids)
    sampler = DistributedSampler(dataset, shuffle=True)
    # pad_to=768 (the largest Chai-1 crop bucket): pins every batch's pair
    # tensor to a FIXED shape instead of the per-batch dynamic max. Without
    # this, host RAM grew unboundedly over training and OOM-killed the job
    # (confirmed 2026-08-27, reproduced at the identical step twice,
    # unaffected by MALLOC_ARENA_MAX) -- PyTorch's pinned-memory allocator
    # (pin_memory=True below) caches a block per distinct tensor size it
    # sees and never releases them, so a varying L_max (this dataset spans
    # 256/384/512/768 depending on which proteins get sampled) keeps
    # growing the cache forever. See collate_fn's docstring for detail.
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_fn, pad_to=768),
    )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    model.train()

    loss_log_path = Path(args.scratch_dir) / "loss_log.csv"
    if global_rank == 0 and not loss_log_path.exists():
        with open(loss_log_path, "w") as f:
            f.write("step,epoch,loss,lr,gate_lr,total_grad_norm,gate_grad_norm\n")

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)

        for (z0, z1), pair, _frame_idx, padding_mask in dataloader:

            # Preemption check, coordinated across ranks. SIGUSR1 arrives
            # asynchronously and independently per rank, so checking only
            # the LOCAL flag is unsafe: a rank that receives it slightly
            # earlier can race ahead into this exit-only dist.barrier()
            # while another rank is still mid-collective on the ORDINARY
            # per-step gradient all-reduce inside loss.backward() below --
            # each then waits on a different collective the other rank
            # never joins, deadlocking until NCCL's watchdog times out
            # (confirmed: job 10246771, rank 1's watchdog timed out after
            # 600s waiting on an ALLREDUCE while rank 0 had already reached
            # this barrier and printed its "Graceful exit" line). This
            # all_reduce is safe to call here specifically because every
            # rank only reaches the top of this loop after finishing the
            # SAME previous step's collectives together -- i.e. all ranks
            # are already mutually synchronized at this exact point,
            # regardless of whether any individual rank's flag was already
            # set moments earlier. It forces a unanimous, synchronized
            # decision before anyone is allowed to diverge into the
            # exit-only path.
            preemption_flag = torch.tensor(
                [1 if PREEMPTION_REQUESTED else 0], device=device, dtype=torch.int32
            )
            dist.all_reduce(preemption_flag, op=dist.ReduceOp.MAX)
            if preemption_flag.item() == 1:
                if global_rank == 0:
                    save_checkpoint(
                        checkpoint_dir, step, epoch,
                        model.module, ema_model, optimizer, scheduler,
                    )
                    print(f"Graceful exit at step {step}.", flush=True)
                dist.barrier()
                dist.destroy_process_group()
                sys.exit(0)

            z0           = z0.to(device,           non_blocking=True)
            z1           = z1.to(device,           non_blocking=True)
            pair         = pair.to(device,         non_blocking=True)
            padding_mask = padding_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = matcher.compute_loss(model, z0, z1, pair, padding_mask)

            loss.backward()

            # Measured pre-clip so gate_grad_norm reflects the natural,
            # unclipped gradient reaching the AdaLN-zero parameters — the
            # signal we actually care about diagnosing, not what's left of
            # it after the global clip has already rescaled everything.
            grad_norms = compute_group_grad_norms(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            update_ema(ema_model, model.module)
            step += 1

            if global_rank == 0 and step % args.log_interval == 0:
                # Param groups are [decay, no_decay, gate] — see get_param_groups.
                lr_now      = scheduler.get_last_lr()[0]
                gate_lr_now = scheduler.get_last_lr()[2]
                loss_val = loss.item()
                print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss_val:.4f} | "
                    f"LR: {lr_now:.2e} | GateLR: {gate_lr_now:.2e} | "
                    f"GradNorm: {grad_norms['total_grad_norm']:.4f} | "
                    f"GateGradNorm: {grad_norms['gate_grad_norm']:.6f}",
                    flush=True,
                )
                with open(loss_log_path, "a") as f:
                    f.write(
                        f"{step},{epoch},{loss_val:.6f},{lr_now:.2e},{gate_lr_now:.2e},"
                        f"{grad_norms['total_grad_norm']:.6f},{grad_norms['gate_grad_norm']:.8f}\n"
                    )

            if (args.checkpoint_interval > 0
                    and step % args.checkpoint_interval == 0
                    and global_rank == 0):
                save_checkpoint(
                    checkpoint_dir, step, epoch,
                    model.module, ema_model, optimizer, scheduler,
                )

            # Diagnostic/short-run cap — checkpoint and exit cleanly once the
            # requested step count is hit, same graceful-exit pattern as the
            # SIGUSR1 preemption handler. Lets a quick instrumentation run
            # (e.g. a few thousand steps) be validated cheaply before
            # committing to a full multi-day run.
            if args.max_steps > 0 and step >= args.max_steps:
                if global_rank == 0:
                    save_checkpoint(
                        checkpoint_dir, step, epoch,
                        model.module, ema_model, optimizer, scheduler,
                    )
                    print(f"Reached --max-steps={args.max_steps}. Exiting at step {step}.",
                          flush=True)
                dist.barrier()
                dist.destroy_process_group()
                sys.exit(0)

        # End-of-epoch checkpoint (rank 0 only)
        if global_rank == 0:
            save_checkpoint(
                checkpoint_dir, step, epoch + 1,
                model.module, ema_model, optimizer, scheduler,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-dir",    type=str,   required=True)
    parser.add_argument("--batch-size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--gate-lr-multiplier", type=float, default=10.0,
                        help="LR multiplier for AdaLN-zero gate/final-projection "
                             "params, which sit behind a self-throttling multiplicative "
                             "bottleneck and need to grow faster than the rest of the "
                             "network to unblock it (default 10x base LR)")
    parser.add_argument("--weight-decay",   type=float, default=1e-5)
    parser.add_argument("--warmup-steps",   type=int,   default=2000)
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--log-interval",   type=int,   default=50)
    parser.add_argument("--proteins",             type=str,   default=None,
                        help="Comma-separated protein IDs to include (default: all)")
    parser.add_argument("--checkpoint-interval", type=int,   default=0,
                        help="Save checkpoint every N steps (0 = off, use for scavenger)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Checkpoint and exit after this many total steps "
                             "(0 = unlimited, run full --epochs). Use for short "
                             "diagnostic runs.")
    args = parser.parse_args()
    main(args)
