"""
training/train_t3.py   (TMLR Phase 3a)

Single-GPU trainer for the controlled t-sampling experiment. The two arms differ in EXACTLY one thing, --t-dist
(uniform vs beta); architecture, data, split, seed, optimizer, schedule, batch size and step count are identical.
Small FluxionDiT (default d_model 256, 6 layers, 8 heads, d_ff 1024 ~ 9M params), centered latents, real-token mask,
held-out PDB entries. See training/phase3_data.py and experiments/EXPERIMENT_LOG.md (Phase 3a pre-registration).

Effective batch 256 by gradient accumulation (micro-batch --batch-size x --accum), warmup + cosine schedule, t CLAMPED to [0.01, 0.99] in both arms,
live flow-loss reduction (1 - loss / loss_of_v=0 on the training batches) logged every --log-interval optimizer steps, and milestone checkpoints at
25/50/75/100% of the run for the intermediate evaluations. (Batch 256 because the E3 gradient-SNR measurement gives SNR = 1 at about 103 samples.)

Checkpoints atomically every --ckpt-interval steps and resumes from <out-dir>/checkpoint.pt, so a 4 h partition cap
costs at most one interval. A monitor line reports the held-out 1-step improvement over identity (learning curve only;
the pre-registered evaluation is scripts/phase3_eval.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from training.dit_model import FluxionDiT
from training.flow_matching import FlowMatcher
from training.phase3_data import PAD_TO, build_datasets, collate, load_pair_store, split_proteins
from training.train import get_param_groups, update_ema


def build_model(a) -> FluxionDiT:
    return FluxionDiT(d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads, d_ff=a.d_ff)


@torch.no_grad()
def one_step_improvement(model, batch, pairs, device, t0: float = 0.01) -> float:
    """100 * (1 - MSE(z0 + v(z0, t0)) / MSE(z0)) over real tokens of a fixed held-out batch (t0 = the lower clip, the smallest t the model is trained at)."""
    z0, z1, pi, mask = (x.to(device) for x in batch)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        v = model(z0, pairs[pi], torch.full((z0.shape[0],), t0, device=device), mask)
    valid = (~mask).float().unsqueeze(-1)
    e_id = (((z1 - z0) ** 2) * valid).sum()
    e_m = (((z1 - (z0 + v.float())) ** 2) * valid).sum()
    return float(100 * (1 - e_m / e_id))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latents-dir", type=Path, default=Path("/scratch/arjunc4/fluxion_data/latents"))
    ap.add_argument("--meta-json", type=Path, default=Path("/scratch/arjunc4/fluxion_data/extraction_metadata.json"))
    ap.add_argument("--mean-dir", type=Path, default=Path("/scratch/arjunc4/fluxion_data/phase3/mean_pos"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--t-dist", choices=["uniform", "beta"], required=True)
    ap.add_argument("--beta-a", type=float, default=0.25)
    ap.add_argument("--beta-b", type=float, default=0.25)
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=8, help="micro-batch per forward/backward")
    ap.add_argument("--accum", type=int, default=32, help="micro-batches per optimizer step (effective batch = batch-size x accum)")
    ap.add_argument("--lr", type=float, default=1e-4 * (256 / 8) ** 0.5, help="peak LR: 1e-4 at batch 8 scaled by sqrt(256/8) = 5.657e-4")
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--t-clip", type=float, nargs=2, default=[0.01, 0.99])
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
    pilot = [p.name for p in sorted(a.latents_dir.iterdir()) if p.is_dir() and (p / "manifest.json").exists()]
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
    tr = build_datasets(a.latents_dir, a.meta_json, a.mean_dir, train_p, index_of, ns)
    ho = build_datasets(a.latents_dir, a.meta_json, a.mean_dir, held_p, index_of, ns)
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
    matcher = FlowMatcher(t_dist=a.t_dist, beta_a=a.beta_a, beta_b=a.beta_b, t_clip=tuple(a.t_clip))
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
    while step < a.max_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            z0, z1, pi, mask = (x.to(dev, non_blocking=True) for x in next(it))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss, v0 = matcher.compute_loss(model, z0, z1, pairs[pi], mask, return_stats=True)
            (loss / a.accum).backward()
            loss_acc += loss.item() / a.accum; v0_acc += v0.item() / a.accum
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
        opt.step(); sched.step(); update_ema(ema, model, decay=a.ema_decay)
        step += 1; n_acc += 1; gn_acc += gn
        if step % a.log_interval == 0:
            mon_ema = mon_raw = float("nan")
            if step % a.monitor_interval == 0:
                model.eval(); mon_raw = one_step_improvement(model, mon, pairs, dev); mon_ema = one_step_improvement(ema, mon, pairs, dev); model.train()
            red = 100 * (1 - loss_acc / v0_acc)                      # live flow-loss reduction on training batches: 1 - L / L(v=0), same t distribution
            w.writerow([step, f"{loss_acc / n_acc:.6f}", f"{v0_acc / n_acc:.6f}", f"{red:.3f}", f"{sched.get_last_lr()[0]:.2e}", f"{gn_acc / n_acc:.5f}", f"{mon_ema:.3f}", f"{mon_raw:.3f}", f"{time.time() - t_start:.0f}"]); logf.flush()
            print(f"[t3] step {step} train loss {loss_acc / n_acc:.5f} (v=0: {v0_acc / n_acc:.5f}) live flow-loss reduction {red:+.2f}% | monitor held-out 1-step vs identity: ema {mon_ema:.2f}% raw {mon_raw:.2f}% ({time.time() - t_start:.0f}s)", flush=True)
            loss_acc = v0_acc = gn_acc = 0.0; n_acc = 0
        if step in milestones:
            save_milestone(milestones[step]); print(f"[t3] milestone {milestones[step]}% saved at step {step}", flush=True)
        if step % a.ckpt_interval == 0:
            save()
        if a.time_limit_min and (time.time() - t_start) / 60 > a.time_limit_min and step < a.max_steps:
            save(); print(f"[t3] time limit reached at step {step}; checkpointed, exiting for resume", flush=True); sys.exit(3)
    save(); print(f"[t3] done at step {step}", flush=True)


if __name__ == "__main__":
    main()
