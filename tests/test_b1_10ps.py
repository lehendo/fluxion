import pytest
pytest.importorskip("MDAnalysis")                      # the script imports MDAnalysis at module level (cluster env)
import torch
import scripts.phase1b_cartesian_10ps as b


def test_streamed_projection_matches_direct_pca(monkeypatch):
    monkeypatch.setattr(b, "N_FRAMES", 103); monkeypatch.setattr(b, "CHUNK", 16)
    g = torch.Generator().manual_seed(0)
    data = {r: torch.randn(103, 30, generator=g) @ torch.randn(30, 30, generator=g) for r in (1, 2, 3)}
    fn = lambda lo, hi, r: data[r][lo:hi]
    fit_idx = list(range(0, 103, 10))
    S = b.project_all(fn, fit_idx, (1, 2, 3), 5)
    from analysis.geometry import pca_scores
    for h in (1, 2, 3):
        tr = [r for r in (1, 2, 3) if r != h]
        ref = pca_scores(torch.cat([data[r][fit_idx] for r in tr]), data[h], 5)
        assert S[h][h].shape == (103, 5) and torch.allclose(S[h][h], ref, atol=1e-4)
