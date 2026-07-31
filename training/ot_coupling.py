"""
training/ot_coupling.py

Minibatch optimal-transport coupling for OT-CFM training (Tong et al. 2023,
"Improving and Generalizing Flow-Based Generative Models with Minibatch
Optimal Transport"). Standard conditional flow matching pairs each z0/z1
example by index within a minibatch — arbitrary, since z0 is independently
sampled noise unrelated to which z1 it lands next to in the batch. This can
produce trajectories that cross/curve a lot in the learned map, since two
"nearby" z_t points may have been interpolated from very different,
unrelated (z0, z1) pairs. The MSE-optimal predictor for such a target is the
CONDITIONAL EXPECTATION over all consistent (z0, z1) pairs, which averages
over contradictory targets and can systematically compress learned variance
or miss distinct conformational modes entirely — the exact failure
signature diagnosed for the static-ensemble model (2026-07): strong RMSF
shape correlation with suppressed magnitude on a small protein, and
generated samples landing outside the real ensemble's occupied conformation
modes on a larger, structurally more complex one.

OT-CFM instead re-pairs z0/z1 within each minibatch via the transport plan
minimizing total squared-distance transport cost, so trajectories cross
less and the regression target is less ambiguous. For uniform minibatch
marginals (equal weight per sample — true here, one sample per batch
index), the discrete OT problem is exactly the linear assignment problem,
solved here via the Hungarian algorithm (scipy) — exact, not an
entropic/Sinkhorn approximation, and trivially cheap at the minibatch sizes
used in this project (16-32).

Note: the codebase has referred to its training objective as "OT-CFM" since
early in this project, but until this module, it implemented plain
conditional flow matching with independent (index-order) pairing — the "OT"
half of the name was aspirational, not actual. This module makes it real.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment


def compute_ot_coupling(
    z0: torch.Tensor,                            # [B, L, D]
    z1: torch.Tensor,                             # [B, L, D]
    padding_mask: torch.Tensor | None = None,     # [B, L] bool, True = padded
) -> torch.Tensor:
    """
    Compute the minibatch-OT-optimal re-pairing of z1 against a FIXED z0.

    Padded positions are excluded from the cost computation on a per-pair
    basis (only positions valid in BOTH sample i and sample j contribute),
    so that batches mixing proteins of different lengths — common here,
    since a training batch is drawn from a ConcatDataset spanning all 19
    proteins and padded to that batch's own max length — don't bias the
    coupling toward samples that merely share similar padding amounts.

    Returns:
        order: int64 [B] index tensor. The caller MUST apply this SAME
        order to every per-sample tensor that has to stay matched to the
        right protein/frame — z1, its pair conditioning, AND its padding
        mask — not just z1 alone, or the model will be trained against a
        mismatched (wrong protein's) conditioning signal. z0 is not
        reordered; it stays fixed as the anchor.
    """
    B = z0.shape[0]
    if B == 1:
        return torch.zeros(1, dtype=torch.long, device=z0.device)

    if padding_mask is not None:
        valid = ~padding_mask                                       # [B, L]
        valid_pair = valid.unsqueeze(1) & valid.unsqueeze(0)         # [B, B, L]
        diff_sq = (z0.unsqueeze(1) - z1.unsqueeze(0)).pow(2).sum(dim=-1)  # [B, B, L]
        n_valid = valid_pair.sum(dim=-1).clamp(min=1)                # [B, B]
        cost = (diff_sq * valid_pair).sum(dim=-1) / n_valid          # [B, B]
    else:
        cost = (z0.unsqueeze(1) - z1.unsqueeze(0)).pow(2).mean(dim=(-1, -2))  # [B, B]

    cost_np = cost.detach().float().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)

    order = torch.empty(B, dtype=torch.long)
    order[torch.from_numpy(row_ind)] = torch.from_numpy(col_ind)
    return order.to(z0.device)
