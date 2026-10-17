#!/usr/bin/env python3
"""
scripts/phase1b_stability_null.py

Chance-corrected PCA-axis stability (pre-registered in experiments/EXPERIMENT_LOG.md, 2026-10-17, before any corrected value was computed). Pure numpy: inputs are the observed statistics
(results/phase1/b1_10ps_stability), the fitted variance spectra (results/phase1/b1_10ps_controls) and the ambient dimensions D_chai = 384 n_res, D_cart = 3 n_res.
Nulls: E[S_m] = m/D (random subspaces; checked by simulation below), E|cos| = Gamma(D/2) / (sqrt(pi) Gamma((D+1)/2)), capture c0 = min(1, (m/D)/f_m) with f_m from the stored top-k spectrum.
kappa = (obs - null) / (1 - null).
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.phase1b_report_10ps import SET13, boot_ci, entry

STAB = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase1/b1_10ps_stability")
CTRL = Path(sys.argv[2] if len(sys.argv) > 2 else "results/phase1/b1_10ps_controls")
LA = Path(sys.argv[3] if len(sys.argv) > 3 else "results/phase1/atlas10ps/local_alignment")
THRESH = 0.05


def null_overlap(m: int, D: int) -> float:
    return min(1.0, m / D)


def null_axis_cos(D: int) -> float:
    return float(np.exp(gammaln(D / 2) - gammaln((D + 1) / 2)) / np.sqrt(np.pi))


def kappa(obs: float, null: float) -> float:
    return float("nan") if null >= 0.999 else (obs - null) / (1 - null)


def capture_null(m: int, D: int, spectrum) -> float:
    f = float(np.cumsum(spectrum)[min(m, len(spectrum)) - 1] / np.sum(spectrum))
    return min(1.0, (m / D) / f)


def simulate_checks() -> None:
    rng = np.random.default_rng(0)
    print("simulation check of the analytic nulls (random subspaces / unit vectors):")
    for D, m in ((138, 100), (700, 100), (20000, 100), (150000, 100)):
        s = []
        for _ in range(3):
            A = np.linalg.qr(rng.standard_normal((D, m)))[0]; B = np.linalg.qr(rng.standard_normal((D, m)))[0]
            s.append(np.mean(np.linalg.svd(A.T @ B, compute_uv=False) ** 2))
        v = rng.standard_normal((200, D)); v /= np.linalg.norm(v, axis=1, keepdims=True); w = rng.standard_normal((200, D)); w /= np.linalg.norm(w, axis=1, keepdims=True)
        print(f"  D = {D:6d}, m = {m}: S_m simulated {np.mean(s):.4f} vs m/D {null_overlap(m, D):.4f};  E|cos| simulated {np.abs((v * w).sum(1)).mean():.4f} vs analytic {null_axis_cos(D):.4f}")


def per_protein(r, ctrl, space):
    D = (384 if space == "chai" else 3) * r["n_res"]; s = r["spaces"][space]; spec = np.array(ctrl["spectrum"][space])
    out = {"D": D}
    n = null_overlap(100, D); out["null_S100"] = n
    out["kS100"] = np.mean([kappa(v, n) for v in s["S_m"]["100"] if v is not None])
    c0 = capture_null(100, D, spec); out["null_C100"] = c0
    out["kC100"] = np.mean([kappa(v, c0) for v in s["capture_m"]["100"] if v is not None])
    ac = np.array(s["axis_cos"])[:, 60:137].mean(); na = null_axis_cos(D); out["null_axis"] = na; out["kaxis"] = kappa(float(ac), na)
    for q in ("0.9", "0.95", "0.99"):
        out[f"kS_q{q}"] = np.mean([kappa(v, null_overlap(mq, D)) for mq, v in s["S_mq"][q]])
        out[f"kC_q{q}"] = np.mean([kappa(c, capture_null(mq, D, spec)) for mq, c in zip(s["mq"][q], s["capture_mq"][q])])
    return out


def verdict(d_s, ci_s, d_c, ci_c, cart_k_median, frac_cart_null_over_half) -> str:
    deficit = d_s >= THRESH and ci_s[0] > 0 and d_c >= THRESH and ci_c[0] > 0
    if deficit:
        return "CARTESIAN DEFICIT (mechanism is back)"
    discriminating = 0.2 <= cart_k_median <= 0.9 and frac_cart_null_over_half < 1 / 3
    if (d_s < THRESH or d_c < THRESH) and discriminating:
        return "NO CARTESIAN DEFICIT, WITH POWER"
    if not discriminating:
        return "UNINFORMATIVE (floor/ceiling for the Cartesian statistic)"
    return "INTERMEDIATE"


def main() -> None:
    simulate_checks()
    S = {Path(f).stem: json.load(open(f)) for f in sorted(glob.glob(str(STAB / "*.json")))}
    C = {Path(f).stem: json.load(open(f)) for f in sorted(glob.glob(str(CTRL / "*.json")))}
    passL = [Path(f).stem for f in sorted(glob.glob(str(LA / "*.json"))) if next(iter(json.load(open(f))["per_protein"].values()))["pass"]]
    sets = {"SET13": [p for p in SET13 if p in S]}
    u = sorted(set(sets["SET13"]) | set(p for p in passL if p in S))
    if set(u) != set(sets["SET13"]):
        sets["UNION"] = u
    for sname, ps in sets.items():
        P = {p: {sp: per_protein(S[p], C[p], sp) for sp in ("chai", "cart")} for p in ps}
        ents = sorted(set(entry(p) for p in ps))
        print(f"\n===== {sname}: {len(ps)} proteins / {len(ents)} entries =====")
        print(f"  ambient D: Chai {np.mean([P[p]['chai']['D'] for p in ps]):.0f} (range {min(P[p]['chai']['D'] for p in ps)}-{max(P[p]['chai']['D'] for p in ps)}), Cart {np.mean([P[p]['cart']['D'] for p in ps]):.0f} (range {min(P[p]['cart']['D'] for p in ps)}-{max(P[p]['cart']['D'] for p in ps)})")
        print(f"  chance levels, mean over proteins: S_100 null Chai {np.mean([P[p]['chai']['null_S100'] for p in ps]):.4f}, Cart {np.mean([P[p]['cart']['null_S100'] for p in ps]):.3f}; capture null Chai {np.mean([P[p]['chai']['null_C100'] for p in ps]):.4f}, Cart {np.mean([P[p]['cart']['null_C100'] for p in ps]):.3f}; E|cos| Chai {np.mean([P[p]['chai']['null_axis'] for p in ps]):.4f}, Cart {np.mean([P[p]['cart']['null_axis'] for p in ps]):.3f}")
        res = {}
        for key, desc in (("kS100", "kappa S_100"), ("kC100", "kappa capture m = 100"), ("kaxis", "kappa matched-axis |cos| idx 61-137"), ("kS_q0.9", "kappa S at m_90%"), ("kS_q0.95", "kappa S at m_95%"), ("kS_q0.99", "kappa S at m_99%"),
                          ("kC_q0.9", "kappa capture at m_90%"), ("kC_q0.95", "kappa capture at m_95%"), ("kC_q0.99", "kappa capture at m_99%")):
            def ent(sp):
                return np.array([np.nanmean([P[p][sp][key] for p in ps if entry(p) == e]) for e in ents])
            ch, ca = ent("chai"), ent("cart"); ok = ~np.isnan(ch) & ~np.isnan(ca); d = (ch - ca)[ok]
            mu, lo, hi = boot_ci(d) if len(d) > 1 else (float("nan"),) * 3; res[key] = (mu, (lo, hi))
            print(f"  {key:9s} {desc:38s} Chai {np.nanmean(ch):+.3f}  Cart {np.nanmean(ca):+.3f}  diff (Chai - Cart) {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]  (n entries {int(ok.sum())})")
        kc = np.array([P[p]["cart"]["kS100"] for p in ps]); frac = float(np.mean([P[p]["cart"]["null_S100"] > 0.5 for p in ps]))
        print(f"  Cartesian kappa_S100 median {np.nanmedian(kc):.3f}; fraction of proteins with a Cartesian null above 0.5 at m = 100: {frac:.2f}")
        print(f"  READING ({sname}): {verdict(res['kS100'][0], res['kS100'][1], res['kC100'][0], res['kC100'][1], float(np.nanmedian(kc)), frac)}")


if __name__ == "__main__":
    main()
