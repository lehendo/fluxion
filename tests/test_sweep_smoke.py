"""End-to-end smoke test of scripts/phase1a_fit.py (sweep + 1C + asymptote) on tiny synthetic caches.
The toy process is a stationary reversible AR(1) per lag with correlation rho_lag, so the closed forms are known."""
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import phase1a_fit as F  # noqa: E402


def test_sweep_matches_ar1_closed_forms(tmp_path, monkeypatch):
    g = torch.Generator().manual_seed(0)
    d, L, rows = 8, 12, 1500
    rho = {1: 0.8, 10: 0.4}
    acf = {}
    for k, p in enumerate(F.PILOT):
        mp = torch.randn(L, d, generator=g)
        for lag, r in rho.items():
            pos = torch.randint(0, L, (rows,), generator=g)
            z0 = torch.randn(rows, d, generator=g)
            z1 = r * z0 + (1 - r * r) ** 0.5 * torch.randn(rows, d, generator=g)
            torch.save({"z0": z0 + mp[pos], "z1": z1 + mp[pos], "pos": pos.to(torch.int16), "dit_sqerr": None,
                        "replica": (torch.arange(rows) % 3).to(torch.int8), "frame": torch.arange(rows, dtype=torch.int32)},
                       tmp_path / f"{p}__lag{lag}.pt")
        torch.save({"mean_pos": mp}, tmp_path / f"{p}__meta.pt")
        acf[p] = {"tau_int_frames": 5.0, "taus": [1, 10], "rho": [rho[1], rho[10]]}
    (tmp_path / "acf.json").write_text(json.dumps(acf))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--cache", str(tmp_path), "--acf", str(tmp_path / "acf.json"), "--out-dir", str(out),
                                      "--lags", "1", "10", "--device", "cpu", "--n-boot", "50"])
    F.main()
    across = json.loads((out / "across_proteins.json").read_text())
    pick = lambda lag, pred: next(a for a in across if a["lag"] == lag and a["predictor"] == pred)
    for lag, r in rho.items():
        # direct linear regression improvement over identity = (1 - rho)/2 ; over the mean = rho^2
        assert abs(pick(lag, "linear_diag")["id_mean"] - 100 * (1 - r) / 2) < 2.0
        assert abs(pick(lag, "linear_diag")["mean_mean"] - 100 * r * r) < 3.0
        # the one-parameter shrinkage model is the exact optimum for this process: full linear cannot beat it by much
        o = json.loads((out / "onec.json").read_text())
        rec = next(x for x in o if x["lag"] == lag)
        assert abs(rec["linear_full.pct_over_shrink"][0]) < 1.0
        assert abs(rec["alpha_mean"] - r) < 0.03
    asym = json.loads((out / "asymptote.json").read_text())
    for a in asym:                                   # measured identity/mean == 2(1-rho) built independently from the ACF
        assert abs(a["measured_minus_from_rho"][0]) < 0.08
