#!/usr/bin/env python3
"""Frame-exact local alignment test of coordinates against latents (paper Appendix A.4)."""
from __future__ import annotations
from fluxion.paths import ATLAS_10PS, DATA_ROOT
import argparse, json, re, sys
from pathlib import Path
import numpy as np

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")
MIN_ALIGNED, MIN_MARGIN, M_OFFSET, N_PAIRS, N_START, WINDOW = 0.20, 0.15, 100, 1000, 9900, 10000


def start_frames(n: int = N_PAIRS, n_start: int = N_START, seed: int = 0) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).choice(n_start, size=n, replace=False))


def shifted_starts(js: np.ndarray, shift: int = WINDOW // 2, n_start: int = N_START) -> np.ndarray:
    """Half-window shift that keeps (j', j'+m) inside the trajectory: circular within the valid start range."""
    return (js + shift) % n_start


def verdict(rho_aligned: float, controls: list[float]) -> dict:
    ctrl = max(controls)
    return {"aligned": rho_aligned, "max_control": ctrl, "margin": rho_aligned - ctrl,
            "pass": bool(rho_aligned >= MIN_ALIGNED and rho_aligned - ctrl >= MIN_MARGIN)}


def main() -> None:
    import MDAnalysis as mda
    import torch
    from scipy.stats import spearmanr
    from fluxion.analysis.geometry import superposed_rmsd
    from fluxion.data_pipeline.dataset import LatentDataset, _REPLICA_RE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-root", type=Path, default=ATLAS_10PS)
    ap.add_argument("--metadata-json", type=Path, default=DATA_ROOT / "extraction_metadata.json")
    ap.add_argument("--latents-dir", type=Path, default=DATA_ROOT / "latents")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "phase1_results/atlas_10ps_local_alignment.json")
    ap.add_argument("--proteins", nargs="+", default=PILOT)
    a = ap.parse_args()
    meta = json.loads(a.metadata_json.read_text())
    ns = torch.load(a.latents_dir / "norm_stats.pt", weights_only=True)
    js, jsh = start_frames(), shifted_starts(start_frames())
    report = {}
    print(f"{'protein':8s} | per replica: aligned / max control (margin) | verdict")
    for p in a.proteins:
        d = a.full_root / p
        if not (d / ".complete").exists():
            report[p] = {"status": "NOT FETCHED", "pass": False}; print(f"{p:8s} NOT FETCHED"); continue
        n_res = meta[p]["n_residues"]
        pdb = sorted(x for x in d.rglob("*.pdb"))[0]
        ds = LatentDataset(a.latents_dir, p, load_pair=False, norm_stats=ns, lag=1)
        sl = {}
        for i, (path, _) in enumerate(ds.index_map):
            c = int(re.sub(r"\D", "", _REPLICA_RE.search(path.name).group(1))); sl.setdefault(c, [i, i + 1])[1] = i + 1

        def lat_dist(rep: int, starts: np.ndarray) -> np.ndarray:
            out = []
            for j in starts:
                z0 = ds._load_frame(sl[rep][0] + int(j))[:n_res].flatten().double()
                z1 = ds._load_frame(sl[rep][0] + int(j) + M_OFFSET)[:n_res].flatten().double()
                out.append(float((z0 - z1).norm()))
            return np.array(out)

        y = {r: lat_dist(r, js) for r in (1, 2, 3)}
        rows = {}
        for r in (1, 2, 3):
            xtc = sorted(d.rglob(f"*R{r}*.xtc"))[0]
            u = mda.Universe(str(pdb), str(xtc)); ca = u.select_atoms("name CA")
            need = sorted(set(js.tolist()) | set((js + M_OFFSET).tolist()))
            pos = {}
            for f in need:
                u.trajectory[f]; pos[f] = ca.positions.copy()
            x = np.array([superposed_rmsd(pos[int(j)], pos[int(j) + M_OFFSET]) for j in js])
            rho = lambda yy: float(spearmanr(x, yy)[0])
            others = [rho(y[q]) for q in (1, 2, 3) if q != r]
            shift = rho(lat_dist(r, jsh))
            rows[f"R{r}"] = {**verdict(rho(y[r]), others + [shift]), "cross_replica": others, "half_window_shift": shift}
        ok = all(v["pass"] for v in rows.values())
        report[p] = {"replicas": rows, "pass": ok}
        print(f"{p:8s} | " + "  ".join(f"R{r}: {v['aligned']:.3f}/{v['max_control']:.3f} ({v['margin']:+.3f})" for r, v in ((r, rows[f'R{r}']) for r in (1, 2, 3))) + f" | {'PASS' if ok else 'FAIL'}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"per_protein": report, "criteria": {"min_aligned": MIN_ALIGNED, "min_margin": MIN_MARGIN, "offset_steps": M_OFFSET, "n_pairs": N_PAIRS}}, indent=1))
    print("PASS:", [q for q, v in report.items() if v["pass"]])


if __name__ == "__main__":
    main()
