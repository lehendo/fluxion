"""
training/phase3_data.py

Data pieces shared by the Phase 3a trainer and evaluator (TMLR conversion; see experiments/EXPERIMENT_LOG.md).

What differs from the original 72 h run, on purpose and identically for BOTH arms of 3a:
  * latents are CENTERED by the protein's own per-position mean (the coordinates every probe in the paper was measured in);
  * the loss/attention mask is the REAL-residue mask derived from sequence length (the original mask only covered positions
    beyond the Chai-1 bucket, so bucket padding counted as ordinary zero-error tokens);
  * held-out PDB entries exist (the original run was in-sample);
  * the 768-bucket protein (1b25_D) is excluded so every sample fits in a 512 pad (pair tensors are the memory bottleneck).
Static pair tensors are held on the GPU once and gathered by protein index, so the DataLoader only moves single-token frames.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data_pipeline.dataset import LatentDataset
from analysis.mask import n_residues, real_mask

PAD_TO = 512
EXCLUDED = ("1b25_D",)                      # 768 bucket
N_HELD_OUT_ENTRIES = 4
SPLIT_SEED = 0


def entry_of(protein: str) -> str:
    return protein.split("_")[0]


def split_proteins(all_proteins: list[str], n_held: int = N_HELD_OUT_ENTRIES, seed: int = SPLIT_SEED):
    """Deterministic split by PDB ENTRY (siblings 1bxy_A/B and 1b2s_E/F stay together). Returns (train, held_out)."""
    eligible = sorted(p for p in all_proteins if p not in EXCLUDED)
    entries = sorted({entry_of(p) for p in eligible})
    held_entries = set(random.Random(seed).sample(entries, n_held))
    return ([p for p in eligible if entry_of(p) not in held_entries], [p for p in eligible if entry_of(p) in held_entries])


def per_position_mean(ds: LatentDataset, stride: int = 10) -> torch.Tensor:
    """Mean over every `stride`-th frame (all replicas) of the NORMALIZED latent, [L_bucket, d]. float32."""
    acc, n = None, 0
    for i in range(0, len(ds), stride):
        x = ds._load_frame(i).double()
        acc = x if acc is None else acc + x
        n += 1
    return (acc / n).float()


class CenteredLagPairs(Dataset):
    """(z0, z1) lag pairs of ONE protein in centered coordinates, valid starts only (no replica-end self-pairs).

    Returns z0, z1 : [L_bucket, d] float32; the protein's index into the GPU pair store; n_res (real residues)."""

    def __init__(self, latents_dir: Path, protein: str, protein_index: int, mean_pos: torch.Tensor, n_res: int,
                 norm_stats: dict, lag: int = 1):
        self.ds = LatentDataset(latents_dir, protein, load_pair=False, norm_stats=norm_stats, lag=lag)
        self.valid = [i for i in range(len(self.ds)) if self.ds._next_idx[i] != i]
        self.mean_pos, self.pi, self.n_res, self.protein = mean_pos, protein_index, n_res, protein

    def __len__(self) -> int:
        return len(self.valid)

    def __getitem__(self, k: int):
        i = self.valid[k]
        z0 = self.ds._load_frame(i) - self.mean_pos
        z1 = self.ds._load_frame(self.ds._next_idx[i]) - self.mean_pos
        return z0, z1, self.pi, self.n_res


def collate(batch, pad_to: int | None = None):
    """Pad singles to pad_to and build the REAL-token mask (True = ignore). Returns z0, z1, protein_idx, mask."""
    pad_to = pad_to or PAD_TO
    z0s, z1s, pis, masks = [], [], [], []
    for z0, z1, pi, n_res in batch:
        L = z0.shape[0]
        z0s.append(F.pad(z0, (0, 0, 0, pad_to - L)))
        z1s.append(F.pad(z1, (0, 0, 0, pad_to - L)))
        pis.append(pi)
        masks.append(torch.arange(pad_to) >= n_res)
    return torch.stack(z0s), torch.stack(z1s), torch.tensor(pis), torch.stack(masks)


def load_pair_store(latents_dir: Path, proteins: list[str], device, pad_to: int | None = None) -> torch.Tensor:
    """[P, pad_to, pad_to, d_pair] on `device`, in the order of `proteins`: bfloat16 on CUDA (the model runs under bf16 autocast), float32 otherwise."""
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    pad_to = pad_to or PAD_TO
    out = []
    for p in proteins:
        pr = torch.load(Path(latents_dir) / p / "pair.pt", weights_only=True).float()
        L = pr.shape[0]
        out.append(F.pad(pr, (0, 0, 0, pad_to - L, 0, pad_to - L)).to(dtype))
    return torch.stack(out).to(device)


def build_datasets(latents_dir: Path, meta_json: Path, mean_dir: Path, proteins: list[str], index_of: dict[str, int], norm_stats: dict):
    out = {}
    for p in proteins:
        mp = torch.load(Path(mean_dir) / f"{p}.pt", weights_only=True)
        out[p] = CenteredLagPairs(latents_dir, p, index_of[p], mp, n_residues(meta_json, p), norm_stats)
    return out
