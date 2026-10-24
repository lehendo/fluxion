"""analysis/template_features.py: the template features through which coordinates enter the frozen Chai-1 trunk, recomputed from backbone coordinates
(chai_lab/data/dataset/templates/load.py): unbinned pairwise pseudo-beta distances and backbone-frame unit vectors. Pure torch."""
from __future__ import annotations
import torch


def template_features(N, CA, C, CB, eps=1e-12):
    """N, CA, C, CB: [B, n, 3] float -> (dist [B, n*n], unit [B, n*n*3]) as torch float32."""
    N, CA, C, CB = (torch.from_numpy(x) for x in (N, CA, C, CB))
    dist = torch.cdist(CB, CB)
    e0 = C - CA; e0 = e0 / (e0.norm(dim=-1, keepdim=True) + eps)
    v = N - CA; v = v - (v * e0).sum(-1, keepdim=True) * e0; e1 = v / (v.norm(dim=-1, keepdim=True) + eps)
    e2 = torch.cross(e0, e1, dim=-1)
    R = torch.stack([e0, e1, e2], dim=-1)                                            # [B, n, 3, 3]: columns are the frame axes
    rel = CA[:, None, :, :] - CA[:, :, None, :]                                       # [B, i, j, 3] = CA_j - CA_i
    loc = torch.einsum("bikc,bijk->bijc", R, rel)                                     # R_i^T (CA_j - CA_i)
    unit = loc / (loc.norm(dim=-1, keepdim=True) + eps)
    return dist.reshape(dist.shape[0], -1), unit.reshape(unit.shape[0], -1)
