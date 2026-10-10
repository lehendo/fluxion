"""
analysis/probes.py

Predictors of z1 used in the identity-relative rescoring. All operate on rows:
each row is one (frame, residue-position) 384-vector, already normalized with the
same norm_stats the DiT uses. Rows are CENTERED by the protein's own per-position
mean before fitting (removes residue identity, which dominates raw variance).

Predictors (all return z1_hat in the same centered coordinates):
  mean      : the training pool's mean of z1 (the OLD baseline; reported only to
              bridge to the earlier 91% / 96% numbers, never as a reference)
  identity  : z1_hat = z0   (zero parameters; the only honest reference)
  linear_diag: per-channel OLS  z1_c ~ a_c + k_c z0_c   (matches the old probe)
  linear_full: z1 ~ A z0 + b                             (needed for Phase 1C)
  mlp       : linear skip + 2 hidden GELU layers
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Closed-form linear
# ---------------------------------------------------------------------------

def fit_linear_diag(z0: torch.Tensor, z1: torch.Tensor, z_prev: torch.Tensor | None = None):
    """
    Per-channel OLS. Without z_prev: z1_c = a_c + k0_c z0_c.
    With z_prev: z1_c = a_c + k0_c z0_c + k1_c zprev_c (the OLD probe's form).
    Returns (a, k0, k1) with k1 = None if z_prev is None. float64 internally.
    """
    z0d, z1d = z0.double(), z1.double()
    n, d = z0d.shape
    if z_prev is None:
        m0, m1 = z0d.mean(0), z1d.mean(0)
        x, y = z0d - m0, z1d - m1
        k0 = (x * y).sum(0) / (x * x).sum(0).clamp_min(1e-30)
        a = m1 - k0 * m0
        return a.float(), k0.float(), None
    zp = z_prev.double()
    a = torch.zeros(d, dtype=torch.float64)
    k0 = torch.zeros(d, dtype=torch.float64)
    k1 = torch.zeros(d, dtype=torch.float64)
    ones = torch.ones(n, dtype=torch.float64)
    for c in range(d):
        X = torch.stack([ones, z0d[:, c], zp[:, c]], dim=1)
        sol = torch.linalg.lstsq(X, z1d[:, c : c + 1]).solution.squeeze(1)
        a[c], k0[c], k1[c] = sol[0], sol[1], sol[2]
    return a.float(), k0.float(), k1.float()


def predict_linear_diag(z0, a, k0, z_prev=None, k1=None):
    out = a.unsqueeze(0) + k0.unsqueeze(0) * z0
    if z_prev is not None:
        out = out + k1.unsqueeze(0) * z_prev
    return out


def fit_linear_full(z0: torch.Tensor, z1: torch.Tensor, ridge: float = 1e-8):
    """z1 ~ A z0 + b. Returns (A [d,d], b [d]) with z1_hat = z0 @ A.T + b."""
    z0d, z1d = z0.double(), z1.double()
    n, d = z0d.shape
    X = torch.cat([z0d, torch.ones(n, 1, dtype=torch.float64)], dim=1)
    G = X.T @ X
    G = G + ridge * torch.trace(G) / (d + 1) * torch.eye(d + 1, dtype=torch.float64)
    W = torch.linalg.solve(G, X.T @ z1d)  # [d+1, d]
    return W[:d].T.contiguous().float(), W[d].float()


def predict_linear_full(z0, A, b):
    return z0 @ A.T + b.unsqueeze(0)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLPProbe(nn.Module):
    """Linear skip + 2-hidden-layer GELU MLP. The skip lets it represent
    identity/linear maps trivially so it is not a capacity-starved straw man."""

    def __init__(self, d: int = 384, hidden: int = 512):
        super().__init__()
        self.skip = nn.Linear(d, d)
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, x):
        return self.skip(x) + self.net(x)


def train_mlp(
    z0_tr, z1_tr, z0_va, z1_va,
    hidden: int = 512, lr: float = 1e-3, batch_size: int = 2048,
    max_epochs: int = 20, patience: int = 3, device: str = "cpu", seed: int = 0,
):
    """
    Early stopping uses (z0_va, z1_va), which MUST be drawn from training-side
    entries only -- never the held-out fold (the old probe selected on its
    test proteins). Returns (model_on_cpu, best_val_mse, n_epochs_run).
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = MLPProbe(z0_tr.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    z0_tr, z1_tr = z0_tr.to(dev), z1_tr.to(dev)
    z0_va, z1_va = z0_va.to(dev), z1_va.to(dev)
    n = z0_tr.shape[0]
    best, best_state, bad, epochs = float("inf"), None, 0, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(z0_tr[idx]), z1_tr[idx])
            loss.backward()
            opt.step()
        epochs += 1
        model.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(model(z0_va), z1_va).item()
        if v < best - 1e-9:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.cpu().eval(), best, epochs


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def row_sqerr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-row squared error averaged over channels -> [N]. Mean of this over
    rows equals the token-and-channel-averaged MSE the flow loss reports."""
    return ((pred - target) ** 2).mean(dim=-1)
