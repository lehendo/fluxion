#!/usr/bin/env python3
"""Gradient signal-to-noise of the original run's checkpoint (paper Section 6.2)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import argparse, json, sys, time
from functools import partial
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from fluxion.data_pipeline.dataset import build_multi_protein_dataset, collate_fn
from fluxion.model.dit import FluxionDiT
from fluxion.training.flow_matching import FlowMatcher

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,"
         "1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")
KS = [1, 4, 16, 64, 256]


def estimates(S, G2, M):
    n2 = float((S.double() ** 2).sum())
    mu2 = (n2 - G2) / (M * (M - 1))
    tr = (G2 - n2 / M) / (M - 1)
    return mu2, tr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch-dir", type=Path, default=DATA_ROOT)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--microbatches", type=int, default=4096)
    ap.add_argument("--mb-size", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "phase1_results/gradsnr.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    ckpt = args.checkpoint or args.scratch_dir / "checkpoints" / "checkpoint.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = FluxionDiT(); model.load_state_dict(state["model"]); model.to(dev).train()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    gate = torch.cat([torch.full((p.numel(),), ("adaLN_mod" in n or "final_layer.linear" in n), dtype=torch.bool) for n, p in params]).to(dev)
    D = int(gate.numel()); print(f"[snr] {D:,} parameters ({int(gate.sum()):,} in the gate/adaLN-zero group); ckpt step {state['step']}", flush=True)
    matcher = FlowMatcher()

    ds = build_multi_protein_dataset(args.scratch_dir / "latents", protein_ids=PILOT)
    M, B = args.microbatches, args.mb_size
    g_idx = torch.Generator().manual_seed(args.seed)
    idx = torch.randint(0, len(ds), (M * B,), generator=g_idx).tolist()
    loader = DataLoader(ds, batch_size=1, sampler=idx, num_workers=args.num_workers, collate_fn=partial(collate_fn, pad_to=None))

    S = torch.zeros(D, device=dev); SS = torch.zeros(D, device=dev)
    chunk_len = M // args.chunks
    S_chunk = [torch.zeros(D, device=dev) for _ in range(args.chunks)]
    G2 = 0.0; G2_gate = 0.0; G2_chunk = [0.0] * args.chunks; G2g_chunk = [0.0] * args.chunks
    A = {k: torch.zeros(D, device=dev) for k in KS if k > 1}; cnt = {k: 0 for k in KS}; emp = {k: [] for k in KS}
    losses = []; gnorms = []
    t0 = time.time(); it = iter(loader)
    for m in range(M):
        model.zero_grad(set_to_none=True); lsum = 0.0
        for _ in range(B):
            (z0, z1), pair, _f, mask = next(it)
            z0, z1, pair, mask = z0.to(dev), z1.to(dev), pair.to(dev), mask.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = matcher.compute_loss(model, z0, z1, pair, mask)
            (loss / B).backward(); lsum += loss.item() / B
        g = torch.cat([p.grad.flatten().float() for _, p in params])
        losses.append(lsum)
        gn2 = float((g.double() ** 2).sum()); gnorms.append(gn2 ** 0.5)
        S += g; SS += g * g; G2 += gn2; G2_gate += float((g[gate].double() ** 2).sum())
        c = min(m // chunk_len, args.chunks - 1)
        S_chunk[c] += g; G2_chunk[c] += gn2; G2g_chunk[c] += float((g[gate].double() ** 2).sum())
        for k in KS:
            if k == 1:
                emp[1].append(gn2); continue
            A[k] += g; cnt[k] += 1
            if cnt[k] == k:
                emp[k].append(float(((A[k] / k).double() ** 2).sum())); A[k].zero_(); cnt[k] = 0
        if (m + 1) % 256 == 0:
            print(f"[snr] {m + 1}/{M} microbatches, {time.time() - t0:.0f}s, mean loss {np.mean(losses):.5f}", flush=True)

    mu2, tr = estimates(S, G2, M)
    mu2g, trg = estimates(S[gate], G2_gate, M)
    # jackknife over chunks (leave one chunk out)
    jk, jk_g = [], []
    for j in range(args.chunks):
        Sj = S - S_chunk[j]; Mj = M - (chunk_len if j < args.chunks - 1 else M - chunk_len * (args.chunks - 1))
        m_, t_ = estimates(Sj, G2 - G2_chunk[j], Mj); mg, tg_ = estimates(Sj[gate], G2_gate - G2g_chunk[j], Mj)
        jk.append((m_, t_)); jk_g.append((mg, tg_))
    def jk_stats(pairs, full):
        mus = np.array([p[0] for p in pairs]); trs = np.array([p[1] for p in pairs]); J = len(pairs)
        se = lambda a: float(np.sqrt((J - 1) / J * ((a - a.mean()) ** 2).sum()))
        kstar = trs / np.where(mus > 0, mus, np.nan)
        return {"mu2_se": se(mus), "tr_se": se(trs), "kstar_leave_one_out": [None if np.isnan(x) else float(x) for x in kstar]}
    mu_j = (S ** 2 - SS) / (M * (M - 1)); var_j = (SS - S ** 2 / M) / (M - 1)
    per_coord = {str(k): float((mu_j > var_j / k).float().mean()) for k in KS}
    res = {"ckpt_step": int(state["step"]), "microbatches": M, "mb_size": B, "n_params": D,
           "mean_loss": float(np.mean(losses)), "loss_std_per_microbatch": float(np.std(losses)),
           "norm_Eg": float(np.sqrt(max(mu2, 0))), "mu2": mu2, "tr_cov": tr, "std_g_per_microbatch": float(np.sqrt(tr)),
           "kstar": (tr / mu2 if mu2 > 0 else None), "snr_k1": float(np.sqrt(max(mu2, 0) / tr)),
           "snr_curve": {str(k): float(np.sqrt(max(mu2, 0) * k / tr)) for k in KS},
           "empirical_mean_sq_norm_avg_of_k": {str(k): float(np.mean(emp[k])) for k in KS},
           "predicted_mean_sq_norm_avg_of_k": {str(k): float(mu2 + tr / k) for k in KS},
           "n_groups": {str(k): len(emp[k]) for k in KS},
           "jackknife": jk_stats(jk, (mu2, tr)),
           "gate_group": {"n_params": int(gate.sum()), "mu2": mu2g, "tr_cov": trg, "kstar": (trg / mu2g if mu2g > 0 else None),
                          "snr_curve": {str(k): float(np.sqrt(max(mu2g, 0) * k / trg)) for k in KS}, "jackknife": jk_stats(jk_g, (mu2g, trg))},
           "fraction_coordinates_with_snr_above_1_at_k": per_coord,
           "k_used_in_training_microbatches": 4, "effective_batch_used": 16,
           "mean_grad_norm_per_microbatch": float(np.mean(gnorms))}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("norm_Eg", "std_g_per_microbatch", "kstar", "snr_k1", "snr_curve", "mean_loss")}, indent=1))
    print("gate group:", json.dumps({k: res["gate_group"][k] for k in ("kstar", "snr_curve")}))
    print("empirical vs predicted mean sq norm of k-averaged gradient:")
    for k in KS:
        print(f"  k={k:4d} (effective batch {k * B:5d}): empirical {res['empirical_mean_sq_norm_avg_of_k'][str(k)]:.4e}  predicted {res['predicted_mean_sq_norm_avg_of_k'][str(k)]:.4e}  ({res['n_groups'][str(k)]} groups)")
    print("fraction of coordinates with per-coordinate SNR>1 at k:", per_coord)


if __name__ == "__main__":
    main()
