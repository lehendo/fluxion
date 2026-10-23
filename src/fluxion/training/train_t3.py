"""
training/train_t3.py

Single-GPU trainer for the controlled t-sampling experiment. The two arms differ in EXACTLY one thing, --t-dist
(uniform vs beta); architecture, data, split, seed, optimizer, schedule, batch size and step count are identical.
Small FluxionDiT (default d_model 256, 6 layers, 8 heads, d_ff 1024 ~ 9M params), centered latents, real-token mask,
held-out PDB entries. See fluxion/training/phase3_data.py and the paper (Section 6).

Effective batch 256 by gradient accumulation (micro-batch --batch-size x --accum), warmup + cosine schedule, t CLAMPED to [0.01, 0.99] in both arms,
live flow-loss reduction (1 - loss / loss_of_v=0 on the training batches) logged every --log-interval optimizer steps, and milestone checkpoints at
25/50/75/100% of the run for the intermediate evaluations. (Batch 256 because the E3 gradient-SNR measurement gives SNR = 1 at about 103 samples.)

IN-RUN DIVERGENCE GUARD (added after the beta arm of the first runs of the controlled experiment diverged unattended): after warmup, if the training loss averaged over the last
--guard-window-samples samples exceeds --guard-ratio x the v=0 loss over the same batches (or any step loss is non-finite), the trainer checkpoints, writes the sentinel
file <out-dir>/HALTED_DIVERGENCE (JSON: step, ratio) and exits with code 4. The sbatch wrapper does not requeue on 4, and a restart refuses to run while the sentinel exists.
Stop conditions never depend on an external poller.

Checkpoints atomically every --ckpt-interval steps and resumes from <out-dir>/checkpoint.pt, so a 4 h partition cap
costs at most one interval. A monitor line reports the held-out 1-step improvement over identity (learning curve only;
the evaluation is scripts/phase3_eval.py).
"""

from __future__ import annotations
from fluxion.paths import DATA_ROOT

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from fluxion.model.dit import FluxionDiT
from fluxion.training.flow_matching import FlowMatcher
from fluxion.training.phase3_data import PAD_TO, build_datasets, collate, load_pair_store, n_real_table, real_mask_from, split_proteins
from fluxion.training.train import get_param_groups, update_ema


def build_model(a) -> FluxionDiT:
    return FluxionDiT(d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads, d_ff=a.d_ff)


@torch.no_grad()
def one_step_improvement(model, batch, pairs, device, t0: float = 0.01, n_real=None) -> float:
    """100 * (1 - MSE(z0 + v(z0, t0)) / MSE(z0)) over real tokens of a fixed held-out batch (t0 = the lower clip, the smallest t the model is trained at)."""
    z0, z1, pi, mask = (x.to(device) for x in batch)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        v = model(z0, pairs[pi], torch.full((z0.shape[0],), t0, device=device), mask)
    smask = real_mask_from(pi, n_real, z0.shape[1]) if n_real is not None else mask           # SCORE on real tokens whatever the training mask is
    valid = (~smask).float().unsqueeze(-1)
    e_id = (((z1 - z0) ** 2) * valid).sum()
    e_m = (((z1 - (z0 + v.float())) ** 2) * valid).sum()
    return float(100 * (1 - e_m / e_id))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latents-dir", type=Path, default=DATA_ROOT / "latents")
    ap.add_argument("--meta-json", type=Path, default=DATA_ROOT / "extraction_metadata.json")
    ap.add_argument("--mean-dir", type=Path, default=DATA_ROOT / "phase3/mean_pos")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--t-dist", choices=["uniform", "beta"], required=True)
    ap.add_argument("--beta-a", type=float, default=0.25)
    ap.add_argument("--beta-b", type=float, default=0.25)
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=8, help="micro-batch per forward/backward")
    ap.add_argument("--accum", type=int, default=32, help="micro-batches per optimizer step (effective batch = batch-size x accum)")
    ap.add_argument("--lr", type=float, default=1e-4 * (256 / 8) ** 0.5, help="peak LR: 1e-4 at batch 8 scaled by sqrt(256/8) = 5.657e-4")
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--guard-ratio", type=float, default=2.0, help="halt if windowed train loss > this x windowed v=0 loss after warmup (0 disables)")
    ap.add_argument("--guard-window-samples", type=int, default=12800)
    ap.add_argument("--guard-mode", choices=["halt", "record"], default="halt", help="halt = stop + sentinel (real runs); record = log the excursion and continue (stability diagnostic only)")
    ap.add_argument("--t-mode", choices=["clamp", "truncate"], default="clamp")
    ap.add_argument("--stop-after", type=int, default=0, help="stop (and write diagnostic_summary.json) after this many optimizer steps while keeping the --max-steps schedule (diagnostic only)")
    ap.add_argument("--t-clip", type=float, nargs=2, default=[0.01, 0.99])
    ap.add_argument("--uncentered", action="store_true", help="T3: do NOT subtract the per-position mean (the original run's coordinates)")
    ap.add_argument("--pad-unmasked", action="store_true", help="T3: mask only positions beyond the Chai-1 bucket, so padding counts as tokens in loss/attention (the original run's behaviour)")
    ap.add_argument("--gate-lr-multiplier", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=350)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--ckpt-interval", type=int, default=250)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--monitor-interval", type=int, default=100)
    ap.add_argument("--time-limit-min", type=float, default=0.0, help="checkpoint and exit cleanly after this many wall minutes (0 = off)")
    ap.add_argument("--proteins", type=str, default=None, help="override the train list (smoke tests only)")
    ap.add_argument("--held-out", type=str, default=None, help="override the held-out list (smoke tests only)")
    a = ap.parse_args()

    t_start = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = a.out_dir / "HALTED_DIVERGENCE"
    if sentinel.exists():
        print(f"[t3] {sentinel} exists: a previous run halted on divergence; refusing to start. Log the fix and remove the sentinel deliberately.", flush=True)
        sys.exit(4)
    pilot = [p.name for p in sorted(a.latents_dir.iterdir()) if p.is_dir() and (p / "manifest.json").exists()]
    if a.proteins and a.held_out:                      # smoke tests only: explicit lists, no split computed
        train_p, held_p = a.proteins.split(","), a.held_out.split(",")
    else:
        train_p, held_p = split_proteins(pilot)
        if a.proteins:
            train_p = a.proteins.split(",")
        if a.held_out:
            held_p = a.held_out.split(",")
    all_p = train_p + held_p
    index_of = {p: i for i, p in enumerate(all_p)}
    ns = torch.load(a.latents_dir / "norm_stats.pt", weights_only=True)
    (a.out_dir / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(a).items()}, "train": train_p, "held_out": held_p}, indent=1))

    pairs = load_pair_store(a.latents_dir, all_p, dev)
    tr = build_datasets(a.latents_dir, a.meta_json, a.mean_dir, train_p, index_of, ns, center=not a.uncentered, mask_real=not a.pad_unmasked)
    ho = build_datasets(a.latents_dir, a.meta_json, a.mean_dir, held_p, index_of, ns, center=not a.uncentered, mask_real=not a.pad_unmasked)
    n_real = n_real_table(tr, ho)
    train_ds = ConcatDataset(list(tr.values()))
    g = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(train_ds, batch_size=a.batch_size, sampler=RandomSampler(train_ds, replacement=True, num_samples=10**9, generator=g),
                        num_workers=a.num_workers, pin_memory=True, drop_last=True, collate_fn=collate, persistent_workers=a.num_workers > 0)
    # fixed held-out monitor batch: 16 pairs per held-out protein
    gm = torch.Generator().manual_seed(123)
    mon = collate([ho[p][int(i)] for p in held_p for i in torch.randint(len(ho[p]), (16,), generator=gm)])

    model = build_model(a).to(dev)
    ema = deepcopy(model).requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[t3] arm={a.t_dist} params={n_params / 1e6:.2f}M train={len(train_p)} proteins ({len(train_ds)} pairs) held-out={held_p}", flush=True)
    opt = torch.optim.AdamW(get_param_groups(model, a.weight_decay, a.lr, a.gate_lr_multiplier), lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
    def lr_factor(st: int) -> float:                    # linear warmup, then cosine to min_lr_frac
        if st < a.warmup_steps:
            return st / max(1, a.warmup_steps)
        prog = (st - a.warmup_steps) / max(1, a.max_steps - a.warmup_steps)
        return a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_factor)
    matcher = FlowMatcher(t_dist=a.t_dist, beta_a=a.beta_a, beta_b=a.beta_b, t_clip=tuple(a.t_clip), t_mode=a.t_mode)
    milestones = {int(round(a.max_steps * q)): int(100 * q) for q in (0.25, 0.5, 0.75, 1.0)}

    step, ck = 0, a.out_dir / "checkpoint.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step = st["step"]
        print(f"[t3] resumed from step {step}", flush=True)

    def save() -> None:
        tmp = ck.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "arch": {"d_model": a.d_model, "n_layers": a.n_layers, "n_heads": a.n_heads, "d_ff": a.d_ff}, "t_dist": a.t_dist,
                    "t_cfg": {"dist": a.t_dist, "beta_a": a.beta_a, "beta_b": a.beta_b, "clip": list(a.t_clip)}}, tmp)
        os.replace(tmp, ck)

    def save_milestone(pct: int) -> None:
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "step": step,
                    "arch": {"d_model": a.d_model, "n_layers": a.n_layers, "n_heads": a.n_heads, "d_ff": a.d_ff}, "t_dist": a.t_dist,
                    "t_cfg": {"dist": a.t_dist, "beta_a": a.beta_a, "beta_b": a.beta_b, "clip": list(a.t_clip)}}, a.out_dir / f"milestone_{pct:03d}.pt")

    log_path = a.out_dir / "loss_log.csv"
    new_log = not log_path.exists()
    logf = open(log_path, "a", newline=""); w = csv.writer(logf)
    if new_log:
        w.writerow(["step", "train_loss", "v0_loss", "flow_reduction_pct", "lr", "grad_norm", "monitor_ema_pct", "monitor_raw_pct", "sec"])
    model.train()
    it = iter(loader)
    loss_acc = v0_acc = gn_acc = 0.0; n_acc = 0
    guard_steps = max(1, a.guard_window_samples // (a.batch_size * a.accum))
    guard_win: deque = deque(maxlen=guard_steps)
    trace, first_over, n_over, max_ratio, nonfinite_seen = [], None, 0, 0.0, False
    while step < a.max_steps:
        opt.zero_grad(set_to_none=True)
        step_loss = step_v0 = 0.0
        for _ in range(a.accum):
            z0, z1, pi, mask = (x.to(dev, non_blocking=True) for x in next(it))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss, v0 = matcher.compute_loss(model, z0, z1, pairs[pi], mask, return_stats=True)
            (loss / a.accum).backward()
            step_loss += loss.item() / a.accum; step_v0 += v0.item() / a.accum
        loss_acc += step_loss; v0_acc += step_v0
        guard_win.append((step_loss, step_v0))
        if a.guard_ratio > 0:
            nonfinite = not math.isfinite(step_loss)
            ratio = (sum(x for x, _ in guard_win) / max(1e-30, sum(y for _, y in guard_win))) if len(guard_win) == guard_steps else float("nan")
            over = nonfinite or (step + 1 > a.warmup_steps and ratio > a.guard_ratio)
            if a.guard_mode == "record":
                nonfinite_seen |= nonfinite
                if step + 1 > a.warmup_steps and math.isfinite(ratio):
                    max_ratio = max(max_ratio, ratio)
                if over:
                    n_over += 1; first_over = first_over or step + 1
            elif over:
                sentinel.write_text(json.dumps({"step": step + 1, "windowed_loss_over_v0": None if nonfinite else ratio, "nonfinite": nonfinite, "guard_ratio": a.guard_ratio,
                                                "window_samples": guard_steps * a.batch_size * a.accum}))
                print(f"[t3] DIVERGENCE GUARD: step {step + 1}, windowed loss / v=0 loss = {ratio:.3f} (limit {a.guard_ratio}), nonfinite={nonfinite}. Halting; sentinel written.", flush=True)
                if not nonfinite:
                    save()
                sys.exit(4)
            trace.append((step + 1, step_loss, step_v0, ratio))
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
        opt.step(); sched.step(); update_ema(ema, model, decay=a.ema_decay)
        step += 1; n_acc += 1; gn_acc += gn
        if step % a.log_interval == 0:
            mon_ema = mon_raw = float("nan")
            if step % a.monitor_interval == 0:
                model.eval(); mon_raw = one_step_improvement(model, mon, pairs, dev, n_real=n_real); mon_ema = one_step_improvement(ema, mon, pairs, dev, n_real=n_real); model.train()
            red = 100 * (1 - loss_acc / v0_acc)                      # live flow-loss reduction on training batches: 1 - L / L(v=0), same t distribution
            w.writerow([step, f"{loss_acc / n_acc:.6f}", f"{v0_acc / n_acc:.6f}", f"{red:.3f}", f"{sched.get_last_lr()[0]:.2e}", f"{gn_acc / n_acc:.5f}", f"{mon_ema:.3f}", f"{mon_raw:.3f}", f"{time.time() - t_start:.0f}"]); logf.flush()
            print(f"[t3] step {step} train loss {loss_acc / n_acc:.5f} (v=0: {v0_acc / n_acc:.5f}) live flow-loss reduction {red:+.2f}% | monitor held-out 1-step vs identity: ema {mon_ema:.2f}% raw {mon_raw:.2f}% ({time.time() - t_start:.0f}s)", flush=True)
            loss_acc = v0_acc = gn_acc = 0.0; n_acc = 0
        if a.stop_after and step >= a.stop_after:
            gn_final = gn
            (a.out_dir / "guard_trace.csv").write_text("step,step_loss,step_v0,windowed_ratio\n" + "\n".join(f"{x},{y:.6f},{z:.6f},{r:.5f}" for x, y, z, r in trace))
            (a.out_dir / "diagnostic_summary.json").write_text(json.dumps({"stop_after": a.stop_after, "seed": a.seed, "t_dist": a.t_dist, "t_mode": a.t_mode, "lr": a.lr, "guard_ratio": a.guard_ratio,
                "window_samples": guard_steps * a.batch_size * a.accum, "max_windowed_ratio_after_warmup": max_ratio, "first_step_over": first_over, "n_steps_over": n_over,
                "diverged": bool(n_over > 0 or nonfinite_seen), "nonfinite": nonfinite_seen}, indent=1))
            save(); print(f"[t3] diagnostic stop at step {step}: max windowed ratio {max_ratio:.3f}, steps over {n_over}", flush=True); sys.exit(0)
        if step in milestones:
            save_milestone(milestones[step]); print(f"[t3] milestone {milestones[step]}% saved at step {step}", flush=True)
        if step % a.ckpt_interval == 0:
            save()
        if a.time_limit_min and (time.time() - t_start) / 60 > a.time_limit_min and step < a.max_steps:
            save(); print(f"[t3] time limit reached at step {step}; checkpointed, exiting for resume", flush=True); sys.exit(3)
    save(); print(f"[t3] done at step {step}", flush=True)


if __name__ == "__main__":
    main()
