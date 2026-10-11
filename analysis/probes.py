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
    ones_n = float(n)
    Sx, Sw, Sy = z0d.sum(0), zp.sum(0), z1d.sum(0)
    Sxx, Sww, Sxw = (z0d * z0d).sum(0), (zp * zp).sum(0), (z0d * zp).sum(0)
    Sxy, Swy = (z0d * z1d).sum(0), (zp * z1d).sum(0)
    G = torch.stack([torch.stack([torch.full_like(Sx, ones_n), Sx, Sw], -1),
                     torch.stack([Sx, Sxx, Sxw], -1),
                     torch.stack([Sw, Sxw, Sww], -1)], -2)                   # [d, 3, 3]
    b = torch.stack([Sy, Sxy, Swy], -1).unsqueeze(-1)                       # [d, 3, 1]
    beta = torch.linalg.solve(G, b).squeeze(-1)                             # [d, 3]
    return beta[:, 0].float(), beta[:, 1].float(), beta[:, 2].float()


RIDGE_GRID = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def fit_ridge_diag(z0: torch.Tensor, z1: torch.Tensor, c: float):
    """Per-channel ridge regression z1_c = a_c + k_c z0_c with a COMMON absolute penalty lambda = c * mean_c(Sxx_c) on the centered slope
    (scale-free in the strength c; c = 0 is plain OLS = fit_linear_diag). Returns (a, k)."""
    z0d, z1d = z0.double(), z1.double()
    m0, m1 = z0d.mean(0), z1d.mean(0)
    x, y = z0d - m0, z1d - m1
    sxx, sxy = (x * x).sum(0), (x * y).sum(0)
    k = sxy / (sxx + c * sxx.mean()).clamp_min(1e-30)
    return (m1 - k * m0).float(), k.float()


def select_ridge_strength(tr0, tr1, va0, va1, grid=RIDGE_GRID) -> float:
    """Pick c by validation MSE on held-in data (never the test fold)."""
    best, best_c = float("inf"), grid[0]
    for c in grid:
        a, k = fit_ridge_diag(tr0, tr1, c)
        mse = float(((a + k * va0 - va1) ** 2).mean())
        if mse < best:
            best, best_c = mse, c
    return best_c


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


# ---------------------------------------------------------------------------
# The OLD probe's MLP, verbatim architecture/optimizer/schedule (scripts/compute_nonlinear_probe.py),
# with the data held on the device. Used only by the Step 0 ladder.
# ---------------------------------------------------------------------------

class OldMLPProbe(nn.Module):
    def __init__(self, d_single: int = 384, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d_single, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_single))

    def forward(self, z_prev, z0):
        return self.net(torch.cat([z0, z_prev], dim=-1))


def train_old_mlp(zp_tr, z0_tr, z1_tr, zp_va, z0_va, z1_va, hidden=256, lr=1e-3, batch_size=1024,
                  max_epochs=30, patience=4, device="cpu", seed=0, val_chunk=100_000):
    """Adam, batch 1024, <=30 epochs, patience 4, early stop on the GIVEN validation rows.
    S0.0 passes the TEST proteins here (the old leak); S0.1+ pass training-side entries.
    Data stays on the CPU (as in the old script) and batches are moved per step, so this fits any GPU
    (a first version held the ~10 GB training set on the device and OOM'd on a 14.5 GB card)."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = OldMLPProbe(z0_tr.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n, best, state, bad, ep = z0_tr.shape[0], float("inf"), None, 0, 0
    nva = z0_va.shape[0]

    def val_mse():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for a in range(0, nva, val_chunk):
                sl = slice(a, a + val_chunk)
                pred = model(zp_va[sl].to(dev), z0_va[sl].to(dev))
                tot += ((pred - z1_va[sl].to(dev)) ** 2).sum().item()
        return tot / (nva * z1_va.shape[1])

    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for s0 in range(0, n, batch_size):
            i = perm[s0:s0 + batch_size]
            opt.zero_grad(set_to_none=True)
            nn.functional.mse_loss(model(zp_tr[i].to(dev), z0_tr[i].to(dev)), z1_tr[i].to(dev)).backward()
            opt.step()
        ep += 1
        v = val_mse()
        if v < best - 1e-8:
            best, bad, state = v, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model.cpu().eval(), best, ep
