"""
tests/test_evaluate_ensemble.py

Unit tests for scripts/evaluate_ensemble.py. CPU-only, synthetic point
clouds. The critical property these tests must verify: the diversity and
RMSF metrics must correctly distinguish a COLLAPSED ensemble (all structures
identical) from a genuinely DIVERSE one — a false negative here (reporting
diversity when there is none) would be a much worse mistake than a false
positive, given this is the primary mode-collapse detector for the whole
static-ensemble pivot.

Run with:
    pytest tests/test_evaluate_ensemble.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_ensemble import (  # noqa: E402
    kabsch_align,
    align_ensemble,
    pairwise_rmsd_distribution,
    compute_rmsf,
    fit_pca,
    project_pca,
    histogram_overlap_js,
    filter_majority_shape,
)


class TestKabschAlign:
    def test_recovers_identical_structure_after_rigid_transform(self):
        rng = np.random.default_rng(0)
        ref = rng.standard_normal((20, 3)).astype(np.float32)

        # Rotate + translate ref by a known rigid transform.
        theta = 0.7
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1],
        ], dtype=np.float32)
        translation = np.array([5.0, -3.0, 2.0], dtype=np.float32)
        transformed = ref @ R.T + translation

        aligned = kabsch_align(transformed, ref)
        assert np.allclose(aligned, ref, atol=1e-4)

    def test_align_ensemble_shapes(self):
        rng = np.random.default_rng(1)
        ref = rng.standard_normal((10, 3)).astype(np.float32)
        coords = rng.standard_normal((5, 10, 3)).astype(np.float32)
        aligned = align_ensemble(coords, ref)
        assert aligned.shape == coords.shape


class TestPairwiseRmsdDistribution:
    def test_zero_for_identical_structures(self):
        """The core mode-collapse-detection correctness check: if every
        ensemble member is the exact same structure, all pairwise RMSDs
        must be ~0."""
        rng = np.random.default_rng(2)
        one_structure = rng.standard_normal((15, 3)).astype(np.float32)
        collapsed = np.stack([one_structure.copy() for _ in range(10)], axis=0)

        rmsds = pairwise_rmsd_distribution(collapsed)
        assert np.allclose(rmsds, 0.0, atol=1e-4)

    def test_nonzero_for_genuinely_diverse_structures(self):
        rng = np.random.default_rng(3)
        diverse = rng.standard_normal((30, 15, 3)).astype(np.float32) * 5.0

        rmsds = pairwise_rmsd_distribution(diverse)
        assert rmsds.mean() > 1.0

    def test_pair_cap_reduces_count(self):
        rng = np.random.default_rng(4)
        coords = rng.standard_normal((20, 10, 3)).astype(np.float32)
        full = pairwise_rmsd_distribution(coords, n_pairs=None)
        capped = pairwise_rmsd_distribution(coords, n_pairs=50)
        assert len(full) == 20 * 19 // 2
        assert len(capped) == 50


class TestComputeRmsf:
    def test_zero_for_identical_structures(self):
        rng = np.random.default_rng(5)
        one_structure = rng.standard_normal((12, 3)).astype(np.float32)
        collapsed = np.stack([one_structure.copy() for _ in range(8)], axis=0)

        rmsf = compute_rmsf(collapsed)
        assert np.allclose(rmsf, 0.0, atol=1e-5)

    def test_matches_manual_calculation_for_controlled_fluctuation(self):
        """One atom fixed, one atom fluctuating with a known per-frame
        offset — RMSF must match the hand-computed expected value exactly."""
        n_frames = 4
        offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)  # mean 0
        coords = np.zeros((n_frames, 2, 3), dtype=np.float32)
        coords[:, 1, 0] = offsets  # second atom moves along x only

        rmsf = compute_rmsf(coords)
        expected_moving_rmsf = np.sqrt((offsets ** 2).mean())
        assert rmsf[0] == pytest.approx(0.0, abs=1e-6)
        assert rmsf[1] == pytest.approx(expected_moving_rmsf, abs=1e-5)


class TestPCA:
    def test_recovers_known_variance_direction(self):
        """Data varying almost entirely along the x-axis of atom 0 should
        have PC1 dominate over PC2 in captured variance."""
        rng = np.random.default_rng(6)
        n_frames = 200
        coords = np.zeros((n_frames, 3, 3), dtype=np.float32)
        coords[:, 0, 0] = rng.standard_normal(n_frames) * 10.0   # big variance
        coords[:, 1, 1] = rng.standard_normal(n_frames) * 0.01   # tiny variance

        mean, components = fit_pca(coords, n_components=2)
        proj = project_pca(coords, mean, components)
        var_pc1 = proj[:, 0].var()
        var_pc2 = proj[:, 1].var()
        assert var_pc1 > 50 * var_pc2

    def test_project_pca_output_shape(self):
        rng = np.random.default_rng(7)
        coords = rng.standard_normal((50, 8, 3)).astype(np.float32)
        mean, components = fit_pca(coords, n_components=2)
        proj = project_pca(coords, mean, components)
        assert proj.shape == (50, 2)


class TestHistogramOverlapJS:
    def test_zero_for_identical_distributions(self):
        rng = np.random.default_rng(8)
        points = rng.standard_normal((500, 2)).astype(np.float32)
        js = histogram_overlap_js(points, points.copy())
        assert js == pytest.approx(0.0, abs=1e-3)

    def test_high_for_disjoint_distributions(self):
        rng = np.random.default_rng(9)
        cluster_a = rng.standard_normal((500, 2)).astype(np.float32) + np.array([0.0, 0.0])
        cluster_b = rng.standard_normal((500, 2)).astype(np.float32) + np.array([100.0, 100.0])
        js = histogram_overlap_js(cluster_a, cluster_b)
        # ln(2) ~ 0.693 is the max JS divergence for disjoint supports
        assert js > 0.6

    def test_intermediate_for_partially_overlapping_distributions(self):
        rng = np.random.default_rng(10)
        cluster_a = rng.standard_normal((500, 2)).astype(np.float32)
        cluster_b = rng.standard_normal((500, 2)).astype(np.float32) + np.array([1.0, 0.0])
        js_partial = histogram_overlap_js(cluster_a, cluster_b)
        js_identical = histogram_overlap_js(cluster_a, cluster_a.copy())
        js_disjoint = histogram_overlap_js(
            cluster_a, cluster_a + np.array([100.0, 100.0], dtype=np.float32)
        )
        assert js_identical < js_partial < js_disjoint


class TestFilterMajorityShape:
    """2026-07: a real run hit np.stack raising on a handful of Chai-1
    decodes coming back with a different atom count (masking artifact on
    unusual conditioning inputs), discarding ~3 hours of already-completed
    diffusion sampling for the other 295/300 samples. This is the fix."""

    def test_all_same_shape_no_exclusions(self):
        rng = np.random.default_rng(0)
        coords_list = [rng.standard_normal((10, 3)).astype(np.float32) for _ in range(5)]
        stacked, n_excluded = filter_majority_shape(coords_list)
        assert stacked.shape == (5, 10, 3)
        assert n_excluded == 0

    def test_minority_shape_excluded_majority_kept(self):
        rng = np.random.default_rng(1)
        majority = [rng.standard_normal((10, 3)).astype(np.float32) for _ in range(8)]
        minority = [rng.standard_normal((7, 3)).astype(np.float32) for _ in range(2)]
        stacked, n_excluded = filter_majority_shape(majority + minority)
        assert stacked.shape == (8, 10, 3)
        assert n_excluded == 2

    def test_multiple_distinct_minority_shapes_all_excluded(self):
        """Matches the real incident: several DIFFERENT off-majority shapes
        (613, 522, 224, 245, 294 vs. a majority of 619) — not just one
        outlier value repeated."""
        rng = np.random.default_rng(2)
        majority = [rng.standard_normal((619, 3)).astype(np.float32) for _ in range(295)]
        outliers = [
            rng.standard_normal((n, 3)).astype(np.float32)
            for n in (613, 522, 224, 245, 294)
        ]
        stacked, n_excluded = filter_majority_shape(majority + outliers)
        assert stacked.shape == (295, 619, 3)
        assert n_excluded == 5

    def test_output_dtype_is_float32(self):
        coords_list = [np.random.randn(5, 3).astype(np.float64) for _ in range(3)]
        stacked, _ = filter_majority_shape(coords_list)
        assert stacked.dtype == np.float32

    def test_tie_picks_one_valid_majority_deterministically(self):
        """With an exact tie, any single consistent choice is acceptable —
        just confirm it doesn't crash and returns internally consistent
        counts (kept + excluded == total)."""
        coords_list = [
            np.zeros((5, 3), dtype=np.float32),
            np.zeros((5, 3), dtype=np.float32),
            np.zeros((6, 3), dtype=np.float32),
            np.zeros((6, 3), dtype=np.float32),
        ]
        stacked, n_excluded = filter_majority_shape(coords_list)
        assert stacked.shape[0] + n_excluded == 4
        assert stacked.shape[0] == 2
