"""Synthetic end-to-end test of scripts/phase1_headline.py (tiny caches in the real file format)."""
import json, sys
from pathlib import Path
import torch
import phase1_headline as H  # noqa: E402


def test_headline_runs_and_f0_equals_direct(tmp_path, monkeypatch):
    g = torch.Generator().manual_seed(0)
    d, L, rows = 6, 10, 800
    for p in H.PILOT:
        pos = torch.randint(0, L, (rows,), generator=g)
        mp = torch.randn(L, d, generator=g)
        z0 = torch.randn(rows, d, generator=g)
        z1 = 0.7 * z0 + (1 - 0.7 ** 2) ** 0.5 * torch.randn(rows, d, generator=g)   # stationary, reversible AR(1)
        zp = z0 + 0.1 * torch.randn(rows, d, generator=g)
        torch.save({"z0": z0 + mp[pos], "z1": z1 + mp[pos], "zprev": zp + mp[pos], "pos": pos.to(torch.int16)},
                   tmp_path / f"{p}__lag1.pt")
        torch.save({"mean_pos": mp}, tmp_path / f"{p}__meta.pt")
    out = tmp_path / "h.json"
    monkeypatch.setattr(sys, "argv", ["x", "--cache", str(tmp_path), "--out", str(out), "--n-boot", "50"])
    H.main()
    r = json.loads(out.read_text())
    assert r["f0_equals_direct0_max_gap"] < 1e-4
    # closed form for a stationary reversible AR(1), rho=0.7: identity MSE 2(1-rho)=0.6, optimal linear MSE 1-rho^2=0.51
    # => improvement over identity = (1-rho)/2 = 15%
    assert abs(r["direct0_mean"] - 0.15) < 0.01
    f = [r["per_t_mean_ci"][str(t)][0] for t in H.T_GRID]
    assert f[0] > f[len(f) // 2]                                      # more explainable at t=0 than at t=1/2
    assert abs(f[len(f) // 2]) < 0.05                                 # vanishes near the midpoint
    assert 2.0 < r["ratio_direct0_over_favg"] < 4.0                # attenuation by the random-t objective
