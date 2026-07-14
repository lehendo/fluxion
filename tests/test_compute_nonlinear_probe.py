"""
tests/test_compute_nonlinear_probe.py

Unit tests for scripts/compute_nonlinear_probe.py.

CPU-only. Run with:

    pytest tests/test_compute_nonlinear_probe.py -v

The critical property this suite must verify, above everything else: the
MLP probe must NOT report spurious "structure" on pure noise via overfitting
(a false positive here would be a much worse mistake than a false negative,
given the stakes riding on this tool's answer) — see
test_mlp_does_not_overfit_pure_noise. Equally, it must be able to detect
real nonlinear structure when present (test_mlp_learns_nonlinear_
relationship_absent_from_linear_fit), otherwise a null result from the tool
would be uninformative rather than reassuring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compute_nonlinear_probe import (  # noqa: E402
    fit_linear_baseline,
    linear_predict,
    train_mlp,
    load_pooled_samples,
)


def _write_protein_with_frames(root: Path, protein: str, frames: np.ndarray) -> None:
    protein_dir = root / protein
    shard_dir = protein_dir / protein
    shard_dir.mkdir(parents=True)
    n, L, d = frames.shape
    with h5py.File(shard_dir / "shard_000000.h5", "w") as f:
        f.create_dataset("single", data=frames.astype(np.float16))
        f.create_dataset("frame_idx", data=np.arange(n, dtype=np.int32))
    manifest = {protein: {"shards": [{
        "path": f"{protein}/shard_000000.h5", "n_frames": n,
        "first_idx": 0, "last_idx": n - 1,
    }]}}
    with open(protein_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
    pair = torch.randn(L, L, 4).half()
    torch.save(pair, protein_dir / "pair.pt")


class TestFitLinearBaseline:
    def test_recovers_known_linear_relationship(self):
        torch.manual_seed(0)
        n, d = 5000, 3
        z_prev = torch.randn(n, d)
        z0 = torch.randn(n, d)
        true_a, true_k0, true_k1 = 0.5, 0.3, -0.2
        noise = torch.randn(n, d) * 0.01
        z1 = true_a + true_k0 * z0 + true_k1 * z_prev + noise

        a, k0, k1 = fit_linear_baseline(z_prev, z0, z1)

        assert torch.allclose(a, torch.full((d,), true_a), atol=0.05)
        assert torch.allclose(k0, torch.full((d,), true_k0), atol=0.05)
        assert torch.allclose(k1, torch.full((d,), true_k1), atol=0.05)

    def test_predict_matches_manual_computation(self):
        torch.manual_seed(1)
        n, d = 100, 2
        z_prev = torch.randn(n, d)
        z0 = torch.randn(n, d)
        a, k0, k1 = torch.tensor([1.0, 2.0]), torch.tensor([0.5, -0.5]), torch.tensor([0.1, 0.2])

        pred = linear_predict(z_prev, z0, a, k0, k1)
        expected = a.unsqueeze(0) + k0.unsqueeze(0) * z0 + k1.unsqueeze(0) * z_prev
        assert torch.allclose(pred, expected)


class TestTrainMLP:
    def test_mlp_learns_nonlinear_relationship_absent_from_linear_fit(self):
        """z1 = |z0| (elementwise). For zero-mean symmetric z0, a linear fit
        finds near-zero slope (Cov(z0, |z0|) ~ 0 for a symmetric distribution)
        — the linear baseline should show almost no reduction from trivial.
        A nonlinear MLP CAN represent |x| and should generalize to held-out
        data, achieving meaningfully lower val MSE than the linear baseline."""
        torch.manual_seed(2)
        d = 4
        n_train, n_val = 8000, 2000

        def make_data(n):
            z_prev = torch.randn(n, d)
            z0 = torch.randn(n, d)
            z1 = z0.abs() + torch.randn(n, d) * 0.05
            return z_prev, z0, z1

        zp_tr, z0_tr, z1_tr = make_data(n_train)
        zp_va, z0_va, z1_va = make_data(n_val)

        a, k0, k1 = fit_linear_baseline(zp_tr, z0_tr, z1_tr)
        linear_val_mse = torch.nn.functional.mse_loss(
            linear_predict(zp_va, z0_va, a, k0, k1), z1_va
        ).item()
        trivial_val_mse = torch.nn.functional.mse_loss(
            z1_tr.mean(dim=0, keepdim=True).expand_as(z1_va), z1_va
        ).item()

        _model, mlp_val_mse, _history = train_mlp(
            zp_tr, z0_tr, z1_tr, zp_va, z0_va, z1_va,
            hidden=32, lr=1e-3, batch_size=256, max_epochs=25, patience=5, seed=0,
        )

        assert linear_val_mse > trivial_val_mse * 0.9, (
            "linear fit should find ~no signal in |z0| (near-zero correlation "
            f"for symmetric data), got linear={linear_val_mse:.4f} vs "
            f"trivial={trivial_val_mse:.4f}"
        )
        assert mlp_val_mse < linear_val_mse * 0.5, (
            f"MLP should clearly beat the linear baseline on genuine nonlinear "
            f"structure, got mlp={mlp_val_mse:.4f} vs linear={linear_val_mse:.4f}"
        )

    def test_mlp_does_not_overfit_pure_noise(self):
        """z1 is fully independent of (z0, z_prev) — pure noise. The MLP's
        HELD-OUT val MSE must not be meaningfully better than the trivial
        baseline; early stopping on val loss must prevent it from reporting
        spurious 'structure' that's really just memorized training noise."""
        torch.manual_seed(3)
        d = 4
        n_train, n_val = 6000, 2000

        z1_var = 1.0  # z1 ~ N(0, 1), independent of everything
        zp_tr, z0_tr = torch.randn(n_train, d), torch.randn(n_train, d)
        z1_tr = torch.randn(n_train, d) * z1_var
        zp_va, z0_va = torch.randn(n_val, d), torch.randn(n_val, d)
        z1_va = torch.randn(n_val, d) * z1_var

        trivial_val_mse = torch.nn.functional.mse_loss(
            z1_tr.mean(dim=0, keepdim=True).expand_as(z1_va), z1_va
        ).item()

        _model, mlp_val_mse, _history = train_mlp(
            zp_tr, z0_tr, z1_tr, zp_va, z0_va, z1_va,
            hidden=64, lr=1e-3, batch_size=256, max_epochs=30, patience=4, seed=0,
        )

        reduction = 1 - mlp_val_mse / trivial_val_mse
        assert reduction < 0.1, (
            f"MLP should NOT show spurious improvement on pure noise (would "
            f"indicate overfitting/false-positive risk), got reduction={reduction:.4f}"
        )

    def test_early_stopping_uses_best_val_checkpoint_not_last_epoch(self):
        """If val loss gets worse after some epoch (overfitting starting),
        the returned model/mse must reflect the BEST epoch, not whatever the
        training loop happened to end on."""
        torch.manual_seed(4)
        d = 4
        n_train, n_val = 4000, 1000
        zp_tr, z0_tr = torch.randn(n_train, d), torch.randn(n_train, d)
        z1_tr = torch.randn(n_train, d)
        zp_va, z0_va = torch.randn(n_val, d), torch.randn(n_val, d)
        z1_va = torch.randn(n_val, d)

        _model, best_val_mse, history = train_mlp(
            zp_tr, z0_tr, z1_tr, zp_va, z0_va, z1_va,
            hidden=64, lr=2e-3, batch_size=128, max_epochs=25, patience=3, seed=0,
        )
        assert best_val_mse == pytest.approx(min(history), abs=1e-6)


class TestLoadPooledSamples:
    def test_pools_across_proteins_and_flattens_positions(self, tmp_path):
        L, d, n = 5, 3, 50
        rng = np.random.default_rng(6)
        frames_a = rng.standard_normal((n, L, d)).astype(np.float32)
        frames_b = rng.standard_normal((n, L, d)).astype(np.float32)
        _write_protein_with_frames(tmp_path, "protA", frames_a)
        _write_protein_with_frames(tmp_path, "protB", frames_b)

        zp, z0, z1 = load_pooled_samples(
            tmp_path, ["protA", "protB"], norm_stats=None,
            lag=1, velocity_lag=1, max_samples_per_protein=10_000, seed=0,
            num_workers=0,
        )
        # n frames -> n-2 valid (non-self-paired) per protein (lag=1,
        # velocity_lag=1: excludes 1 head + 1 tail), each flattened over L.
        expected_per_protein = (n - 2) * L
        assert zp.shape == (2 * expected_per_protein, d)
        assert z0.shape == (2 * expected_per_protein, d)
        assert z1.shape == (2 * expected_per_protein, d)

    def test_subsampling_cap_enforced(self, tmp_path):
        """Subsampling uses per-sample Bernoulli thinning INSIDE the
        streaming loop (bounded memory — see load_pooled_samples docstring
        for why: a naive accumulate-then-cap approach can hit tens of GB for
        a single real protein before ever subsampling). This means the kept
        count has statistical variance rather than landing on the cap
        exactly, but a final hard trim guarantees it never EXCEEDS the cap.
        Use a large enough dataset that the expected-vs-cap relative
        variance is small, so a loose (not exact) bound is still reliable."""
        L, d, n = 5, 3, 2000
        rng = np.random.default_rng(7)
        frames = rng.standard_normal((n, L, d)).astype(np.float32)
        _write_protein_with_frames(tmp_path, "protCap", frames)

        cap = 200  # total candidates ~= (n-2)*L ~= 9990; keep_prob ~= 0.02
        zp, z0, z1 = load_pooled_samples(
            tmp_path, ["protCap"], norm_stats=None,
            lag=1, velocity_lag=1, max_samples_per_protein=cap, seed=0,
            num_workers=0,
        )
        assert zp.shape[0] <= cap, "final trim must never let count exceed the cap"
        assert zp.shape[0] > cap * 0.5, (
            f"thinning should land reasonably close to the cap ({cap}), got {zp.shape[0]}"
        )
        assert z0.shape[0] == zp.shape[0]
        assert z1.shape[0] == zp.shape[0]
