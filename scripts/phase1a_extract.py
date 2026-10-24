#!/usr/bin/env python3
"""Builds the row cache of real-token pairs behind the lag sweep and probes (paper Sections 3.3 and 4.1)."""

from __future__ import annotations
from fluxion.paths import DATA_ROOT

import argparse
import re
import sys
import zlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from fluxion.data_pipeline.dataset import LatentDataset, _REPLICA_RE
from fluxion.analysis.mask import n_residues, real_mask, verify_mask

DEFAULT_LAGS = (1, 10, 100, 1000, 5000)


def collate(batch):
    out_z, fidx = [], []
    for item, _pair, f in batch:
        out_z.append(torch.stack(list(item)))        # [2 or 3, L, d]
        fidx.append(f)
    return torch.stack(out_z), torch.tensor(fidx)    # [B, 2|3, L, d]


def replica_code(ds: LatentDataset, flat_idx: int) -> int:
    m = _REPLICA_RE.search(ds.index_map[flat_idx][0].name)
    return int(re.sub(r"\D", "", m.group(1))) if m else 0


@torch.no_grad()
def integrate(model, z0, pair, steps):
    """Euler integration of the learned velocity field from t=0 to t=1."""
    B = z0.shape[0]
    pair_b = pair.unsqueeze(0).expand(B, -1, -1, -1)
    z = z0
    for k in range(steps):
        t = torch.full((B,), k / steps, device=z0.device)
        z = z + model(z, pair_b, t, None) / steps
    return z


def load_dit(ckpt_path: Path, device):
    from fluxion.model.dit import FluxionDiT
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    models = {}
    for name, key in (("raw", "model"), ("ema", "ema")):
        m = FluxionDiT()
        m.load_state_dict(state[key])
        models[name] = m.to(device).eval()
    return models, int(state["step"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latents-dir", required=True, type=Path)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--lags", type=int, nargs="+", default=list(DEFAULT_LAGS))
    ap.add_argument("--frames-per-lag", type=int, default=600)
    ap.add_argument("--rows-per-frame", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-positions", action="store_true",
                    help="sample rows uniformly over ALL L positions (incl. bucket padding), flagging is_pad; "
                         "reproduces the old probes' pooling for the Step 0 ladder")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sequences-json", type=Path,
                    default=DATA_ROOT / "extraction_metadata.json")
    ap.add_argument("--dit-checkpoint", type=Path, default=None)
    ap.add_argument("--dit-steps", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ns = torch.load(args.latents_dir / "norm_stats.pt", weights_only=True)
    dev = torch.device(args.device)

    models, ckpt_step, pair = None, None, None
    if args.dit_checkpoint is not None:
        models, ckpt_step = load_dit(args.dit_checkpoint, dev)
        pair = LatentDataset(args.latents_dir, args.protein, load_pair=True,
                             norm_stats=ns, lag=1).static_pair.to(dev)
        print(f"[extract] DiT loaded (step {ckpt_step}), pair {tuple(pair.shape)}", flush=True)

    # --- real-position mask: DERIVED FROM SEQUENCE LENGTH; variance only asserts it ---
    ds_probe = LatentDataset(args.latents_dir, args.protein, load_pair=False, norm_stats=ns, lag=1)
    L = ds_probe._load_frame(0).shape[0]
    n_res = n_residues(args.sequences_json, args.protein)
    real = real_mask(L, n_res)
    # padding is constant only within ONE shard/run, so assert on 24 frames of the first shard
    mask_info = verify_mask(real, torch.stack([ds_probe._load_frame(i) for i in range(24)]))
    real_idx = real.nonzero().squeeze(1)
    n_real = int(real.sum())
    print(f"[extract] {args.protein}: L_bucket={L}, n_residues(sequence)={n_res}, "
          f"padded={mask_info['n_padded']} ({100 * mask_info['n_padded'] / L:.1f}%), "
          f"mask verified constant; real positions with zero variance: "
          f"{mask_info['n_real_with_zero_variance']}", flush=True)
    d = ds_probe._load_frame(0).shape[-1]
    sum_pos = torch.zeros(L, d, dtype=torch.float64)
    cnt = 0

    for lag in args.lags:
        out_path = args.out_dir / f"{args.protein}__lag{lag}.pt"
        ds = LatentDataset(args.latents_dir, args.protein, load_pair=False, norm_stats=ns,
                           lag=lag, velocity_lag=1 if lag == 1 else 0)
        valid = [i for i in range(len(ds))
                 if ds._next_idx[i] != i and (lag != 1 or ds._prev_idx[i] != i)]
        if len(valid) < args.frames_per_lag:
            print(f"[extract] lag {lag}: only {len(valid)} valid starts "
                  f"(< {args.frames_per_lag}); using all", flush=True)
        g = torch.Generator().manual_seed(
            args.seed + zlib.crc32(f"{args.protein}:{lag}".encode()) % 100000)
        pick = sorted(torch.randperm(len(valid), generator=g)[: args.frames_per_lag].tolist())
        flat = [valid[i] for i in pick]
        rep = torch.tensor([replica_code(ds, i) for i in flat])
        loader = DataLoader(Subset(ds, flat), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)

        cols = {k: [] for k in ("z0", "z1", "zprev", "pos", "frame", "replica")}
        dit_err = {f"{n}_n{s}": [] for n in (models or {}) for s in args.dit_steps}
        cursor = 0
        pool_n = L if args.all_positions else n_real
        pool_idx = torch.arange(L) if args.all_positions else real_idx
        P = min(args.rows_per_frame, pool_n)
        for zs, fidx in loader:                              # zs [B, 2|3, L, d]
            B = zs.shape[0]
            z0, z1 = zs[:, -2], zs[:, -1]
            sum_pos += z0.double().sum(0) + z1.double().sum(0)
            cnt += 2 * B
            # rows: P random REAL positions per frame
            sel = pool_idx[torch.rand(B, pool_n, generator=g).argsort(1)[:, :P]]   # [B, P]
            gather = lambda t: torch.gather(t, 1, sel.unsqueeze(-1).expand(-1, -1, d))
            cols["z0"].append(gather(z0).reshape(-1, d))
            cols["z1"].append(gather(z1).reshape(-1, d))
            if lag == 1:
                cols["zprev"].append(gather(zs[:, 0]).reshape(-1, d))
            cols["pos"].append(sel.reshape(-1).to(torch.int16))
            cols["frame"].append(fidx.repeat_interleave(P).to(torch.int32))
            cols["replica"].append(rep[cursor:cursor + B].repeat_interleave(P).to(torch.int8))
            if models is not None:
                mb = 8 if L <= 256 else 4 if L <= 512 else 2
                for a in range(0, B, mb):
                    z0g = z0[a:a + mb].to(dev)
                    z1g = z1[a:a + mb].to(dev)
                    sg = sel[a:a + mb].to(dev)
                    for name, model in models.items():
                        for steps in args.dit_steps:
                            zh = integrate(model, z0g, pair, steps)
                            gi = lambda t: torch.gather(t, 1, sg.unsqueeze(-1).expand(-1, -1, d))
                            e = ((gi(zh) - gi(z1g)) ** 2).mean(-1)       # [mb, P]
                            dit_err[f"{name}_n{steps}"].append(e.reshape(-1).cpu())
            cursor += B

        cell = {
            "protein": args.protein, "lag": lag, "L_bucket": L, "n_real": n_real,
            "n_frames": len(flat), "rows_per_frame": P,
            "all_positions": bool(args.all_positions),
            "z0": torch.cat(cols["z0"]), "z1": torch.cat(cols["z1"]),
            "zprev": torch.cat(cols["zprev"]) if cols["zprev"] else None,
            "pos": torch.cat(cols["pos"]), "is_pad": ~real[torch.cat(cols["pos"]).long()], "frame": torch.cat(cols["frame"]),
            "replica": torch.cat(cols["replica"]),
            "dit_sqerr": {k: torch.cat(v) for k, v in dit_err.items()} or None,
            "dit_ckpt_step": ckpt_step,
        }
        torch.save(cell, out_path)
        ident = ((cell["z1"] - cell["z0"]) ** 2).mean().item()
        msg = f"[extract] lag {lag:5d}: {cell['z0'].shape[0]} rows, identity MSE (real tokens) = {ident:.6f}"
        if cell["dit_sqerr"]:
            msg += "  DiT: " + ", ".join(f"{k}={v.mean().item():.6f}" for k, v in cell["dit_sqerr"].items())
        print(msg, flush=True)

    torch.save({"protein": args.protein, "L_bucket": L, "n_real": n_real, "real_mask": real,
                "mean_pos": (sum_pos / cnt).float(), "n_frames_for_mean": cnt,
                "mask_info": mask_info, "mask_source": "sequence_length"},
               args.out_dir / f"{args.protein}__meta.pt")
    print("[extract] done", flush=True)


if __name__ == "__main__":
    main()
