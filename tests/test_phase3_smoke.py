"""
tests/test_phase3_smoke.py

End-to-end CPU smoke test for the controlled-experiment pipeline (centering, real-token mask, train, resume, evaluate) on synthetic latents.
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

from fluxion.training import phase3_data
from fluxion.training.phase3_data import CenteredLagPairs, collate, entry_of, per_position_mean, split_proteins
from fluxion.data_pipeline.dataset import LatentDataset

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


class TestArmMasses:
    def test_masses_sum_to_one_and_have_clamp_point_masses(self):
        import numpy as np
        import phase3_eval as ev
        for cfg, end in ((dict(dist="uniform", clip=[0.01, 0.99]), 0.01), (dict(dist="beta", beta_a=0.25, beta_b=0.25, clip=[0.01, 0.99]), 0.17)):
            m = ev.arm_masses(cfg)
            assert abs(m.sum() - 1) < 1e-9 and m[0] > end and abs(m[0] - m[-1]) < 1e-9
        # attained reduction of a model that matches the closed-form curve must reproduce the prediction (same integral as scripts/phase3_predict.py)
        from fluxion.analysis.theory import f_closed_form
        from phase3_predict import clamp_expect, ARMS
        for name, cfg in (("uniform", dict(dist="uniform", clip=[0.01, 0.99])), ("beta(0.25,0.25)", dict(dist="beta", beta_a=0.25, beta_b=0.25, clip=[0.01, 0.99]))):
            f = lambda x: f_closed_form(np.asarray(x, dtype=float), 0.695)
            grid_val = float(ev.arm_masses(cfg) @ f(ev.T_GRID))
            assert abs(grid_val - clamp_expect(ARMS[name], f, np.linspace(0.01, 0.99, 4001))) < 2e-3


class TestEndToEnd:
    def test_train_resume_eval(self, tmp_path, monkeypatch):
        monkeypatch.setattr(phase3_data, "PAD_TO", 16)
        proteins = ["1aaa_A", "1bbb_A", "1ccc_A", "1ddd_A"]
        lat = make_store(tmp_path, proteins)
        meansd = tmp_path / "mean"; meansd.mkdir()
        ns = torch.load(lat / "norm_stats.pt", weights_only=True)
        for p in proteins:
            torch.save(per_position_mean(LatentDataset(lat, p, load_pair=False, norm_stats=ns, lag=1), 1), meansd / f"{p}.pt")
        import fluxion.training.train_t3 as t3
        out = {}
        for arm in ("uniform", "beta"):
            o = tmp_path / arm; out[arm] = o
            argv = ["train_t3", "--latents-dir", str(lat), "--meta-json", str(tmp_path / "meta.json"), "--mean-dir", str(meansd), "--out-dir", str(o),
                    "--t-dist", arm, "--max-steps", "6", "--batch-size", "2", "--accum", "2", "--d-model", "32", "--n-layers", "1", "--n-heads", "2", "--d-ff", "64",
                    "--num-workers", "0", "--log-interval", "2", "--monitor-interval", "4", "--ckpt-interval", "3", "--warmup-steps", "2",
                    "--proteins", "1aaa_A,1bbb_A,1ccc_A", "--held-out", "1ddd_A"]
            monkeypatch.setattr(sys, "argv", argv)
            t3.main()
            assert (o / "checkpoint.pt").exists() and torch.load(o / "checkpoint.pt", weights_only=False)["step"] == 6
            assert sorted(q.name for q in o.glob("milestone_*.pt")) == ["milestone_025.pt", "milestone_050.pt", "milestone_075.pt", "milestone_100.pt"]
            assert torch.load(o / "milestone_100.pt", weights_only=False)["t_cfg"]["clip"] == [0.01, 0.99]
            log = (o / "loss_log.csv").read_text().splitlines()
            assert log[0].startswith("step,train_loss,v0_loss,flow_reduction_pct") and len(log) >= 4
        # resume: re-running with the same max-steps is a no-op that keeps the step
        monkeypatch.setattr(sys, "argv", argv)
        t3.main()
        assert torch.load(out["beta"] / "checkpoint.pt", weights_only=False)["step"] == 6
        import phase3_eval as ev
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
        fs = r["flow_summary"]["held_out"]
        assert {"uniform_ema", "beta_ema", "uniform_raw", "beta_raw", "ratio_beta_over_uniform_ema"} <= set(fs)
        assert fs["ratio_beta_over_uniform_ema"]["point"] > 0 and len(r["t_grid"]) == 41
        assert 0.55 < r["alpha"] < 0.85
        assert r["summary"]["in_sample"]["shrink"][0] > 5


class TestDivergenceGuard:
    def _argv(self, tmp_path, out, extra):
        proteins = ["1aaa_A", "1bbb_A"]
        lat = make_store(tmp_path, proteins)
        meansd = tmp_path / "mean"; meansd.mkdir()
        ns = torch.load(lat / "norm_stats.pt", weights_only=True)
        for p in proteins:
            torch.save(per_position_mean(LatentDataset(lat, p, load_pair=False, norm_stats=ns, lag=1), 1), meansd / f"{p}.pt")
        return ["train_t3", "--latents-dir", str(lat), "--meta-json", str(tmp_path / "meta.json"), "--mean-dir", str(meansd), "--out-dir", str(out), "--t-dist", "uniform",
                "--max-steps", "8", "--batch-size", "2", "--accum", "1", "--d-model", "32", "--n-layers", "1", "--n-heads", "2", "--d-ff", "64", "--num-workers", "0",
                "--log-interval", "2", "--monitor-interval", "1000", "--warmup-steps", "2", "--proteins", "1aaa_A", "--held-out", "1bbb_A"] + extra

    def test_guard_halts_writes_sentinel_and_blocks_restart(self, tmp_path, monkeypatch):
        monkeypatch.setattr(phase3_data, "PAD_TO", 16)
        import fluxion.training.train_t3 as t3
        out = tmp_path / "g"
        # a ratio limit of 1e-4 is exceeded by any model that is not near-perfect: the guard must trip right after warmup
        monkeypatch.setattr(sys, "argv", self._argv(tmp_path, out, ["--guard-ratio", "0.0001", "--guard-window-samples", "4"]))
        with pytest.raises(SystemExit) as e:
            t3.main()
        assert e.value.code == 4
        info = json.loads((out / "HALTED_DIVERGENCE").read_text())
        assert info["step"] >= 3 and info["windowed_loss_over_v0"] > 0.0001
        with pytest.raises(SystemExit) as e2:                         # a restart refuses while the sentinel exists
            t3.main()
        assert e2.value.code == 4

    def test_guard_silent_on_a_healthy_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(phase3_data, "PAD_TO", 16)
        import fluxion.training.train_t3 as t3
        out = tmp_path / "h"
        monkeypatch.setattr(sys, "argv", self._argv(tmp_path, out, ["--guard-ratio", "50", "--guard-window-samples", "4"]))
        t3.main()
        assert not (out / "HALTED_DIVERGENCE").exists() and torch.load(out / "checkpoint.pt", weights_only=False)["step"] == 8

    def test_guard_can_be_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(phase3_data, "PAD_TO", 16)
        import fluxion.training.train_t3 as t3
        out = tmp_path / "d"
        monkeypatch.setattr(sys, "argv", self._argv(tmp_path, out, ["--guard-ratio", "0"]))
        t3.main()
        assert not (out / "HALTED_DIVERGENCE").exists()


class TestDiagnosticMode:
    def test_record_mode_continues_and_summarises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(phase3_data, "PAD_TO", 16)
        import fluxion.training.train_t3 as t3
        helper = TestDivergenceGuard()
        out = tmp_path / "diag"
        argv = helper._argv(tmp_path, out, ["--guard-ratio", "0.0001", "--guard-window-samples", "4", "--guard-mode", "record", "--stop-after", "6", "--t-dist", "beta", "--t-mode", "truncate"])
        argv[argv.index("--max-steps") + 1] = "20"
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as e:
            t3.main()
        assert e.value.code == 0                                                     # record mode does NOT halt with code 4
        assert not (out / "HALTED_DIVERGENCE").exists()
        s = json.loads((out / "diagnostic_summary.json").read_text())
        assert s["stop_after"] == 6 and s["diverged"] is True and s["n_steps_over"] >= 1 and s["t_mode"] == "truncate"
        assert len((out / "guard_trace.csv").read_text().splitlines()) == 7        # header + 6 steps


class TestEvalResume:
    def test_partial_resume_and_early_heldout_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(phase3_data, "PAD_TO", 16)
        import fluxion.training.train_t3 as t3
        import phase3_eval as ev
        helper = TestDivergenceGuard()
        proteins = ["1aaa_A", "1bbb_A"]
        out = tmp_path / "arm"
        argv = helper._argv(tmp_path, out, ["--guard-ratio", "0"])
        monkeypatch.setattr(sys, "argv", argv); t3.main()
        lat = tmp_path / "latents"; meansd = tmp_path / "mean"
        monkeypatch.setattr(ev, "N_FRAMES", 4); monkeypatch.setattr(ev, "N_FRAMES_IN_SAMPLE", 4)
        res = tmp_path / "res.json"
        cmd = ["phase3_eval", "--latents-dir", str(lat), "--meta-json", str(tmp_path / "meta.json"), "--mean-dir", str(meansd), "--arm", "uniform", str(out / "checkpoint.pt"), "--out", str(res)]
        monkeypatch.setattr(sys, "argv", cmd); ev.main()
        full = json.loads(res.read_text()); part = res.with_suffix(".partial.json")
        assert part.exists() and set(json.loads(part.read_text())["per_protein"]) == {"held_out:1bbb_A", "in_sample:1aaa_A"}
        # simulate a kill after the held-out group: keep only held-out in the partial file, delete the final JSON, then resume
        d = json.loads(part.read_text()); d["per_protein"] = {k: v for k, v in d["per_protein"].items() if k.startswith("held_out")}; d["flow"] = {k: v for k, v in d["flow"].items() if k.startswith("held_out")}
        part.write_text(json.dumps(d)); res.unlink()
        monkeypatch.setattr(sys, "argv", cmd); ev.main()
        again = json.loads(res.read_text())
        assert set(again["per_protein"]) == set(full["per_protein"])
        assert again["per_protein"]["held_out:1bbb_A"]["pct_vs_identity"] == full["per_protein"]["held_out:1bbb_A"]["pct_vs_identity"]     # resumed protein not recomputed


class TestT3Modes:
    """T3: arms trained uncentered and/or with padding counted must still be SCORED on real tokens, in the same units as the centered/masked arms."""

    def _stores(self, tmp_path):
        proteins = ["1aaa_A", "1bbb_A"]
        lat = make_store(tmp_path, proteins)
        meansd = tmp_path / "mean"; meansd.mkdir()
        ns = torch.load(lat / "norm_stats.pt", weights_only=True)
        for p in proteins:
            torch.save(per_position_mean(LatentDataset(lat, p, load_pair=False, norm_stats=ns, lag=1), 1), meansd / f"{p}.pt")
        return lat, meansd, ns

    def test_dataset_flags(self, tmp_path, monkeypatch):
        from fluxion.training.phase3_data import CenteredLagPairs, collate, n_real_table, real_mask_from
        monkeypatch.setattr(phase3_data, "PAD_TO", 24)
        lat, meansd, ns = self._stores(tmp_path)
        mp = torch.load(meansd / "1aaa_A.pt", weights_only=True)
        base = CenteredLagPairs(lat, "1aaa_A", 0, mp, N_RES, ns)
        unc = CenteredLagPairs(lat, "1aaa_A", 0, mp, N_RES, ns, center=False, mask_real=False)
        z0c, z1c, _, n_c = base[0]; z0u, z1u, _, n_u = unc[0]
        assert n_c == N_RES and n_u == L_BUCKET and unc.n_res == N_RES                       # model mask covers the bucket only; real count preserved for scoring
        assert torch.allclose(z0u - z0c, mp, atol=1e-5) and torch.allclose(z1u - z1c, mp, atol=1e-5)   # same frames, shifted by exactly the per-position mean
        z0, z1, pi, mask = collate([unc[0]])
        assert (~mask[0]).sum() == L_BUCKET and mask[0, L_BUCKET:].all()                    # padding tokens are 'valid' to the model
        smask = real_mask_from(pi, n_real_table({"a": unc}), 24)
        assert (~smask[0]).sum() == N_RES                                                    # but the scoring mask is the real one

    def test_identity_error_is_independent_of_training_coordinates_and_mask(self, tmp_path, monkeypatch):
        import fluxion.training.train_t3 as t3
        import phase3_eval as ev
        monkeypatch.setattr(phase3_data, "PAD_TO", 16)
        helper = TestDivergenceGuard()
        base = helper._argv(tmp_path, tmp_path / "c", ["--guard-ratio", "0"])            # builds the synthetic store once
        base[base.index("--max-steps") + 1] = "4"
        lat, meansd = tmp_path / "latents", tmp_path / "mean"
        res = {}
        for tag, extra in (("c", []), ("u", ["--uncentered", "--pad-unmasked"])):
            out = tmp_path / tag
            argv = list(base); argv[argv.index("--out-dir") + 1] = str(out); argv += extra
            monkeypatch.setattr(sys, "argv", argv); t3.main()
            cfg = json.loads((out / "config.json").read_text())
            assert (cfg["uncentered"] == "True") == (tag == "u") and (cfg["pad_unmasked"] == "True") == (tag == "u")
            monkeypatch.setattr(ev, "N_FRAMES", 6); monkeypatch.setattr(ev, "N_FRAMES_IN_SAMPLE", 6)
            outj = tmp_path / f"{tag}.json"
            cmd = ["phase3_eval", "--latents-dir", str(lat), "--meta-json", str(tmp_path / "meta.json"), "--mean-dir", str(meansd), "--arm", tag, str(out / "checkpoint.pt"), "--out", str(outj), "--skip-flow"]
            if tag == "u":
                cmd += ["--no-baselines"]
            monkeypatch.setattr(sys, "argv", cmd); ev.main()
            res[tag] = json.loads(outj.read_text())
        for k in res["c"]["per_protein"]:
            a, b = res["c"]["per_protein"][k]["mse"]["identity"], res["u"]["per_protein"][k]["mse"]["identity"]
            assert abs(a - b) <= 1e-4 * abs(a), (k, a, b)                                    # same real-token identity MSE in both coordinate systems
        assert "shrink" in res["c"]["per_protein"]["held_out:1bbb_A"]["mse"] and "shrink" not in res["u"]["per_protein"]["held_out:1bbb_A"]["mse"]
        assert res["u"]["alpha"] is None

    def test_uncentered_arm_requires_no_baselines_flag(self, tmp_path, monkeypatch):
        import fluxion.training.train_t3 as t3
        import phase3_eval as ev
        monkeypatch.setattr(phase3_data, "PAD_TO", 16)
        helper = TestDivergenceGuard()
        out = tmp_path / "u"
        argv = helper._argv(tmp_path, out, ["--guard-ratio", "0", "--uncentered"]); argv[argv.index("--max-steps") + 1] = "2"
        monkeypatch.setattr(sys, "argv", argv); t3.main()
        monkeypatch.setattr(sys, "argv", ["phase3_eval", "--latents-dir", str(tmp_path / "latents"), "--meta-json", str(tmp_path / "meta.json"), "--mean-dir", str(tmp_path / "mean"), "--arm", "u", str(out / "checkpoint.pt"), "--out", str(tmp_path / "x.json")])
        with pytest.raises(SystemExit, match="no-baselines"):
            ev.main()
