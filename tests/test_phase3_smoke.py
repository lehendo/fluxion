"""
tests/test_phase3_smoke.py

End-to-end CPU smoke test for the Phase 3a pipeline (centering, real-token mask, train, resume, evaluate) on synthetic latents.
The synthetic process is a stationary reversible AR(1) in each channel so a trained model CAN improve over identity; the test only
checks plumbing and invariants (shapes, masks, centering, determinism of the split, resume, eval protocol), not learning quality.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from training import phase3_data
from training.phase3_data import CenteredLagPairs, collate, entry_of, per_position_mean, split_proteins
from data_pipeline.dataset import LatentDataset

D_S, D_P, L_BUCKET, N_RES, N_FR = 384, 256, 16, 11, 24


def make_store(root: Path, proteins: list[str], seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    lat = root / "latents"
    for p in proteins:
        d = lat / p / p
        d.mkdir(parents=True)
        x = np.zeros((N_FR, L_BUCKET, D_S), dtype=np.float32)
        x[0] = rng.standard_normal((L_BUCKET, D_S))
        for i in range(1, N_FR):
            x[i] = 0.7 * x[i - 1] + np.sqrt(1 - 0.49) * rng.standard_normal((L_BUCKET, D_S))
        x[:, N_RES:, :] = 3.14                                       # bucket padding: constant
        x += rng.standard_normal((1, L_BUCKET, D_S)).astype(np.float32) * 2          # per-position offset (residue identity)
        with h5py.File(d / "shard_000000.h5", "w") as f:
            f.create_dataset("single", data=x.astype(np.float16)); f.create_dataset("frame_idx", data=np.arange(N_FR, dtype=np.int32))
        (lat / p / "manifest.json").write_text(json.dumps({p: {"shards": [{"path": f"{p}/shard_000000.h5", "n_frames": N_FR, "first_idx": 0, "last_idx": N_FR - 1}]}}))
        torch.save(torch.randn(L_BUCKET, L_BUCKET, D_P).half(), lat / p / "pair.pt")
    torch.save({"mean": torch.zeros(D_S), "std": torch.ones(D_S)}, lat / "norm_stats.pt")
    (root / "meta.json").write_text(json.dumps({p: {"n_residues": N_RES} for p in proteins}))
    return lat


class TestSplit:
    PILOT = "16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B".split(",")

    def test_deterministic_and_entry_disjoint(self):
        tr, ho = split_proteins(self.PILOT)
        assert (tr, ho) == split_proteins(list(reversed(self.PILOT)))
        assert not {entry_of(p) for p in tr} & {entry_of(p) for p in ho}
        assert "1b25_D" not in tr + ho and len({entry_of(p) for p in ho}) == 4

    def test_siblings_stay_together(self):
        tr, ho = split_proteins(self.PILOT)
        for a, b in (("1bxy_A", "1bxy_B"), ("1b2s_E", "1b2s_F")):
            assert (a in tr) == (b in tr)


class TestData:
    def test_centering_mask_and_collate(self, tmp_path, monkeypatch):
        lat = make_store(tmp_path, ["p1_A"])
        ns = torch.load(lat / "norm_stats.pt", weights_only=True)
        ds = LatentDataset(lat, "p1_A", load_pair=False, norm_stats=ns, lag=1)
        mp = per_position_mean(ds, stride=1)
        cp = CenteredLagPairs(lat, "p1_A", 0, mp, N_RES, ns)
        assert len(cp) == N_FR - 1                                           # replica-end self-pair excluded
        z0, z1, pi, n = cp[0]
        assert z0.shape == (L_BUCKET, D_S) and n == N_RES
        assert torch.allclose(torch.stack([cp[i][0] for i in range(len(cp))]).mean(0)[:N_RES], mp[:N_RES] * 0, atol=0.6)   # roughly centered
        monkeypatch.setattr(phase3_data, "PAD_TO", 24)
        zz0, zz1, pis, mask = collate([cp[0], cp[1]])
        assert zz0.shape == (2, 24, D_S) and mask.shape == (2, 24)
        assert (~mask[0]).sum() == N_RES and mask[0, N_RES:].all()          # REAL-token mask, not bucket mask
        assert (zz0[:, L_BUCKET:] == 0).all()


class TestEndToEnd:
    def test_train_resume_eval(self, tmp_path, monkeypatch):
        monkeypatch.setattr(phase3_data, "PAD_TO", 16)
        proteins = ["1aaa_A", "1bbb_A", "1ccc_A", "1ddd_A"]
        lat = make_store(tmp_path, proteins)
        meansd = tmp_path / "mean"; meansd.mkdir()
        ns = torch.load(lat / "norm_stats.pt", weights_only=True)
        for p in proteins:
            torch.save(per_position_mean(LatentDataset(lat, p, load_pair=False, norm_stats=ns, lag=1), 1), meansd / f"{p}.pt")
        import training.train_t3 as t3
        out = {}
        for arm in ("uniform", "beta"):
            o = tmp_path / arm; out[arm] = o
            argv = ["train_t3", "--latents-dir", str(lat), "--meta-json", str(tmp_path / "meta.json"), "--mean-dir", str(meansd), "--out-dir", str(o),
                    "--t-dist", arm, "--max-steps", "6", "--batch-size", "2", "--d-model", "32", "--n-layers", "1", "--n-heads", "2", "--d-ff", "64",
                    "--num-workers", "0", "--log-interval", "2", "--monitor-interval", "4", "--ckpt-interval", "3", "--warmup-steps", "2",
                    "--proteins", "1aaa_A,1bbb_A,1ccc_A", "--held-out", "1ddd_A"]
            monkeypatch.setattr(sys, "argv", argv)
            t3.main()
            assert (o / "checkpoint.pt").exists() and torch.load(o / "checkpoint.pt", weights_only=False)["step"] == 6
        # resume: re-running with the same max-steps is a no-op that keeps the step
        monkeypatch.setattr(sys, "argv", argv)
        t3.main()
        assert torch.load(out["beta"] / "checkpoint.pt", weights_only=False)["step"] == 6
        import scripts.phase3_eval as ev
        res = tmp_path / "res.json"
        monkeypatch.setattr(ev, "N_FRAMES", 6)
        monkeypatch.setattr(sys, "argv", ["phase3_eval", "--latents-dir", str(lat), "--meta-json", str(tmp_path / "meta.json"), "--mean-dir", str(meansd),
                                          "--arm", "uniform", str(out["uniform"] / "checkpoint.pt"), "--arm", "beta", str(out["beta"] / "checkpoint.pt"), "--out", str(res)])
        ev.main()
        r = json.loads(res.read_text())
        assert r["summary"]["held_out"]["_n_entries"] == 1 and r["summary"]["in_sample"]["_n_entries"] == 3
        for k in ("identity",):
            assert k not in r["summary"]["held_out"]                              # identity is the reference, not a row
        assert {"shrink", "linear_diag", "dit_uniform_ema_n1", "dit_beta_raw_n8"} <= set(r["summary"]["held_out"])
        # on a stationary AR(1) with rho = 0.7 the fitted shrinkage must beat identity and alpha must be near 0.7
        assert 0.55 < r["alpha"] < 0.85
        assert r["summary"]["in_sample"]["shrink"][0] > 5
