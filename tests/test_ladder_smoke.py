"""End-to-end smoke test of scripts/phase1a_ladder.py on tiny synthetic caches in the real file format."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import phase1a_ladder as ladder  # noqa: E402


def _make_cache(tmp: Path):
    g = torch.Generator().manual_seed(0)
    d, L, rows = 8, 12, 1200
    acf = {}
    for k, p in enumerate(ladder.PILOT):
        n_res = 6 + (k % 5)
        pos = torch.randint(0, L, (rows,), generator=g)
        mp = torch.randn(L, d, generator=g)
        z0 = torch.randn(rows, d, generator=g)
        zp = z0 + 0.1 * torch.randn(rows, d, generator=g)
        z1 = 0.9 * z0 + 0.05 * torch.randn(rows, d, generator=g)
        torch.save({"L_bucket": L, "n_real": n_res, "z0": z0 + mp[pos], "z1": z1 + mp[pos], "zprev": zp + mp[pos],
                    "pos": pos.to(torch.int16), "is_pad": pos >= n_res,
                    "replica": (torch.arange(rows) % 3).to(torch.int8),
                    "frame": torch.arange(rows, dtype=torch.int32)}, tmp / f"{p}__lag1.pt")
        torch.save({"mean_pos": mp}, tmp / f"{p}__meta.pt")
        acf[p] = {"tau_int_frames": 5.0}
    (tmp / "acf.json").write_text(json.dumps(acf))


def test_ladder_runs_end_to_end(tmp_path, monkeypatch, capsys):
    _make_cache(tmp_path)
    out = tmp_path / "ladder.json"
    monkeypatch.setattr(sys, "argv", ["x", "--cache", str(tmp_path), "--acf", str(tmp_path / "acf.json"),
                                      "--out", str(out), "--device", "cpu", "--seeds", "0",
                                      "--n-boot", "20", "--force-rungs"])
    ladder.main()
    res = json.loads(out.read_text())
    assert [r["rung"] for r in res["rungs"]] == ["S0.0", "S0.1", "S0.2", "S0.3", "S0.3-all19"]
    for r in res["rungs"]:
        assert r["identity"] > 0 and r["mean"] > r["identity"]       # identity beats the mean baseline here
    # masking padding must change the numbers (padding rows are in the cache)
    assert res["rungs"][2]["identity"] != res["rungs"][1]["identity"]
    assert "S0.3_across" in res and "S0.0" in res and len(res["S0.0"]["checks"]) == 4
