"""
training/dit_model.py

FluxionDiT — Diffusion Transformer for protein conformational dynamics.

I/O contract
------------
    z    [B, L, d_s=384]     noisy single-token representation at time t
    pair [B, L, L, d_p=256]  clean pair-token representation (structural conditioning)
    t    [B]                  continuous diffusion timestep ∈ [0, 1]
    ──►  v [B, L, d_s=384]   predicted velocity field  v_θ(z, pair, t)

Dimension map
-------------
    d_s     = 384   Chai-1 single-rep channels  (confirmed from test run)
    d_p     = 256   Chai-1 pair-rep channels
    d_model = 384   DiT hidden dim  (== d_s → input_proj is square, no information loss)
    n_heads = 8     attention heads → d_head = 48
    d_ff    = 1536  SwiGLU inner dim  (4× expansion; SwiGLU has 3 weight matrices so
                    effective FLOP ratio vs. standard 2-layer FFN is ~1.5×)
    d_freq  = 256   sinusoidal frequencies for timestep encoding

Architecture
------------
    1. TimestepEmbedding  scalar t → sinusoidal [B, d_freq] → MLP → [B, d_model]
    2. input_proj         [B, L, d_s] → [B, L, d_model]          (square linear)
    3. n_layers × DiTBlock
         ├─ AdaLN-Zero    6 modulation scalars from t_emb        (init to 0 → identity)
         ├─ PairBiasedAttention
         │    self-attn on z  +  pair → [B, L, L, n_heads] bias injected into logits
         └─ SwiGLU FFN
    4. FinalLayer         AdaLN + zero-init linear → v [B, L, d_s]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Maps scalar t → sinusoidal frequency tensor [B, d_freq]."""

    def __init__(self, d_freq: int = 256) -> None:
        super().__init__()
        if d_freq % 2 != 0:
            raise ValueError(f"d_freq must be even, got {d_freq}")
        self.d_freq = d_freq

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]  float, range [0, 1]
        half = self.d_freq // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
        )  # [half]
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, d_freq]


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal encoding → 2-layer MLP → conditioning vector [B, d_model].

    d_freq  → Linear → SiLU → Linear(d_model)
    """

    def __init__(self, d_model: int, d_freq: int = 256) -> None:
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(d_freq)
        self.mlp = nn.Sequential(
            nn.Linear(d_freq, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] → [B, d_model]
        return self.mlp(self.sinusoidal(t))


# ---------------------------------------------------------------------------
# AdaLN helper
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift.  Broadcasts over sequence dim."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Attention with pair-derived bias
# ---------------------------------------------------------------------------

class PairBiasedAttention(nn.Module):
    """
    Multi-head self-attention on single reps with Evoformer-style pair bias.

    The pair representation [B, L, L, d_p] is projected to per-head scalar
    biases [B, n_heads, L, L] and added to the pre-softmax attention logits.
    This lets geometry encoded in the pair rep directly modulate which residues
    attend to which — the same mechanism that drives Evoformer communication.

    Dimensions:
        QKV projection:  d_model → 3 × d_model
        Pair bias:       d_p=256 → n_heads=8  (one scalar bias per head per pair)
        Output:          d_model → d_model
    """

    def __init__(self, d_model: int, d_pair: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # scale is handled internally by SDPA

        self.to_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.pair_bias_proj = nn.Linear(d_pair, n_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        pair: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x:            [B, L, d_model]
        # pair:         [B, L, L, d_pair]
        # padding_mask: [B, L] bool, True = padded token
        B, L, _ = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Pair bias: [B, L, L, d_pair] → [B, n_heads, L, L]
        bias = self.pair_bias_proj(pair).permute(0, 3, 1, 2)

        # Merge padding mask into the additive bias so padded key positions
        # receive -inf before softmax regardless of the pair values.
        if padding_mask is not None:
            # [B, L] → [B, 1, 1, L]: -inf on padded keys, 0 elsewhere
            key_mask = torch.zeros_like(padding_mask, dtype=bias.dtype)
            key_mask = key_mask.masked_fill(padding_mask, float("-inf"))
            bias = bias + key_mask.unsqueeze(1).unsqueeze(2)

        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=bias,
            dropout_p=0.0,
            is_causal=False,
        )
        # When every key in a row is masked the softmax is 0/0 → NaN.
        # Replace with zero: a fully-masked query attends to nothing.
        out = torch.nan_to_num(out, nan=0.0)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)  # [B, L, d_model]
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Feed-forward network (SwiGLU)
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward:  x → SiLU(gate(x)) ⊙ value(x) → proj → output.

    Two d_model → d_ff projections (gate + value) and one d_ff → d_model.
    d_ff = 4 × d_model provides strong expressivity; SwiGLU's multiplicative
    gating typically outperforms GELU MLP at the same parameter count.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff)
        self.value = nn.Linear(d_model, d_ff)
        self.proj = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.gate(x)) * self.value(x))


# ---------------------------------------------------------------------------
# DiT block
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """
    Single Diffusion Transformer block.

    Conditioning flow:
        t_emb [B, d_model]
          └─ SiLU → Linear(d_model, 6·d_model) → split into 6 vectors
               (shift_msa, scale_msa, gate_msa,  — modulate pre-attn norm
                shift_mlp, scale_mlp, gate_mlp)  — modulate pre-ffn norm

    AdaLN-Zero: the modulation linear is zero-initialized so every block
    starts as an identity map, stabilizing early training.
    """

    def __init__(self, d_model: int, d_pair: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = PairBiasedAttention(d_model, d_pair, n_heads)
        self.ff = SwiGLUFFN(d_model, d_ff)

        # AdaLN-Zero: zero-initialize so output is zero at init → blocks are
        # initially bypassed (residual stream sees identity).
        self.adaLN_mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)

    def forward(
        self,
        x: torch.Tensor,                           # [B, L, d_model]
        pair: torch.Tensor,                        # [B, L, L, d_pair]
        t_emb: torch.Tensor,                       # [B, d_model]
        padding_mask: torch.Tensor | None = None,  # [B, L] bool
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_mod(t_emb).chunk(6, dim=-1)  # each [B, d_model]

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), pair, padding_mask
        )
        x = x + gate_mlp.unsqueeze(1) * self.ff(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


# ---------------------------------------------------------------------------
# Terminal output layer
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """
    AdaLN + zero-initialized linear → velocity field [B, L, d_out].

    Zero-initializing the output projection means the model predicts zero
    velocity at t=0, which is a stable starting point for flow matching.
    """

    def __init__(self, d_model: int, d_out: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaLN_mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)

        self.linear = nn.Linear(d_model, d_out)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_mod(t_emb).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class FluxionDiT(nn.Module):
    """
    Fluxion Diffusion Transformer.

    Default configuration matches the Chai-1 latent dimensions confirmed by
    the successful test run (single: [512, 384], pair: [512, 512, 256]).

    Args:
        d_single:  Chai-1 single-rep channels  (default 384)
        d_pair:    Chai-1 pair-rep channels     (default 256)
        d_model:   DiT hidden dim               (default 384, == d_single)
        n_layers:  Number of DiT blocks         (default 12)
        n_heads:   Attention heads              (default 8)
        d_ff:      SwiGLU inner dim             (default 1536 = 4 × d_model)
        d_freq:    Sinusoidal freq dim          (default 256)
    """

    def __init__(
        self,
        d_single: int = 384,
        d_pair: int = 256,
        d_model: int = 384,
        n_layers: int = 12,
        n_heads: int = 8,
        d_ff: int = 1536,
        d_freq: int = 256,
    ) -> None:
        super().__init__()
        self.d_single = d_single
        self.d_model = d_model

        # Square projection so d_single == d_model never discards information.
        # If you change d_model, this becomes a learned bottleneck/expansion.
        self.input_proj = nn.Linear(d_single, d_model)

        self.time_embed = TimestepEmbedding(d_model, d_freq)

        self.blocks = nn.ModuleList([
            DiTBlock(d_model, d_pair, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.final_layer = FinalLayer(d_model, d_single)

    def forward(
        self,
        z: torch.Tensor,                           # [B, L, d_single]
        pair: torch.Tensor,                        # [B, L, L, d_pair]
        t: torch.Tensor,                           # [B]
        padding_mask: torch.Tensor | None = None,  # [B, L] bool, True = padded
    ) -> torch.Tensor:
        """
        Returns velocity field v_θ(z, pair, t): [B, L, d_single].

        padding_mask marks positions that are zero-padding (from batching proteins
        of different lengths). Padded positions are masked out of attention and
        excluded from the loss in FlowMatcher.
        """
        x = self.input_proj(z)        # [B, L, d_model]
        t_emb = self.time_embed(t)    # [B, d_model]

        for block in self.blocks:
            x = block(x, pair, t_emb, padding_mask)

        return self.final_layer(x, t_emb)  # [B, L, d_single]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def param_count(self) -> str:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{n / 1_000_000:.1f}M"

    def __repr__(self) -> str:
        cfg = (
            f"d_single={self.d_single}, d_model={self.d_model}, "
            f"layers={len(self.blocks)}"
        )
        return f"FluxionDiT({cfg}, params={self.param_count()})"
