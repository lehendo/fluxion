import math
import torch
from fluxion.analysis.template_features import template_features


def _backbone(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    ca = torch.cumsum(torch.randn(1, n, 3, generator=g) * 1.5, 1)
    N = ca + torch.randn(1, n, 3, generator=g) * 0.4 + torch.tensor([1.0, 0, 0]); C = ca + torch.randn(1, n, 3, generator=g) * 0.4 + torch.tensor([0, 1.0, 0]); CB = ca + torch.randn(1, n, 3, generator=g) * 0.6
    return N.numpy(), ca.numpy(), C.numpy(), CB.numpy()


def _rot(a, b, c):
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    Rz = torch.tensor([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]]); Ry = torch.tensor([[cb, 0, sb], [0, 1.0, 0], [-sb, 0, cb]]); Rx = torch.tensor([[1.0, 0, 0], [0, cc, -sc], [0, sc, cc]])
    return Rz @ Ry @ Rx


def test_rotation_and_translation_invariance():
    X = _backbone(); d0, u0 = template_features(*X)
    R = _rot(0.7, -0.4, 1.1).numpy(); t = torch.tensor([3.0, -2.0, 9.0]).numpy()
    Y = [x @ R.T + t for x in X]; d1, u1 = template_features(*Y)
    assert torch.allclose(d0, d1, atol=1e-4) and torch.allclose(u0, u1, atol=1e-4)


def test_shapes_unit_norm_and_diagonal():
    X = _backbone(n=6); d, u = template_features(*X)
    assert d.shape == (1, 36) and u.shape == (1, 108)
    U = u.reshape(1, 6, 6, 3); norms = U.norm(dim=-1)
    assert torch.allclose(norms[0][~torch.eye(6, dtype=torch.bool)], torch.ones(30), atol=1e-4) and norms[0].diagonal().abs().max() < 1e-4
    D = d.reshape(1, 6, 6); assert torch.allclose(D[0], D[0].T, atol=1e-5) and D[0].diagonal().abs().max() < 1e-5


def test_features_respond_to_a_side_chain_move_that_calpha_does_not_see():
    X = list(_backbone()); d0, u0 = template_features(*X)
    X[3] = X[3].copy(); X[3][0, 2] += 1.0                                              # move one C-beta only
    d1, u1 = template_features(*X)
    assert (d1 - d0).abs().max() > 0.1
