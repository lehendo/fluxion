"""
tests/test_ot_coupling.py

Unit tests for training/ot_coupling.py. CPU-only. The critical properties
this suite must verify: the returned order is always a valid permutation,
the coupling genuinely reduces total transport cost relative to the naive
identity pairing (not just "a" valid permutation), it picks the objectively
correct pairing on a hand-computable case, and — the specific bug this
module exists to avoid — padded-region noise must not bias the coupling
away from the pairing that's actually optimal on real (valid) content.

Run with:
    pytest tests/test_ot_coupling.py -v
"""

from __future__ import annotations

import torch
import pytest

from training.ot_coupling import compute_ot_coupling


def _total_cost(z0: torch.Tensor, z1: torch.Tensor, order: torch.Tensor) -> float:
    z1_paired = z1[order]
    return ((z0 - z1_paired) ** 2).sum().item()


class TestValidPermutation:
    def test_order_is_a_permutation(self):
        torch.manual_seed(0)
        B, L, D = 8, 5, 4
        z0 = torch.randn(B, L, D)
        z1 = torch.randn(B, L, D)
        order = compute_ot_coupling(z0, z1)
        assert sorted(order.tolist()) == list(range(B))

    def test_order_dtype_and_device(self):
        z0 = torch.randn(4, 3, 2)
        z1 = torch.randn(4, 3, 2)
        order = compute_ot_coupling(z0, z1)
        assert order.dtype == torch.int64
        assert order.device == z0.device

    def test_b_equals_one_trivial(self):
        z0 = torch.randn(1, 5, 4)
        z1 = torch.randn(1, 5, 4)
        order = compute_ot_coupling(z0, z1)
        assert order.tolist() == [0]


class TestReducesTransportCost:
    def test_ot_cost_never_exceeds_identity_cost(self):
        """By definition, the OT-optimal pairing must have total cost <=
        any other permutation, including the naive index-order (identity)
        pairing flow matching used before this module existed."""
        torch.manual_seed(1)
        B, L, D = 16, 10, 8
        z0 = torch.randn(B, L, D)
        z1 = torch.randn(B, L, D)

        identity_order = torch.arange(B)
        ot_order = compute_ot_coupling(z0, z1)

        identity_cost = _total_cost(z0, z1, identity_order)
        ot_cost = _total_cost(z0, z1, ot_order)
        assert ot_cost <= identity_cost + 1e-4

    def test_ot_cost_strictly_lower_when_identity_is_clearly_suboptimal(self):
        """Construct z0/z1 where the identity pairing is deliberately bad
        (each z0[i] is actually closest to a DIFFERENT z1 than z1[i]) —
        confirms compute_ot_coupling finds a genuinely better pairing, not
        just a tied/degenerate one."""
        z0 = torch.tensor([[[0.0]], [[10.0]], [[20.0]]])   # [3, 1, 1]
        z1 = torch.tensor([[[21.0]], [[1.0]], [[11.0]]])   # [3, 1, 1]

        identity_cost = _total_cost(z0, z1, torch.arange(3))
        order = compute_ot_coupling(z0, z1)
        ot_cost = _total_cost(z0, z1, order)
        assert ot_cost < identity_cost


class TestHandComputableOptimum:
    def test_picks_the_objectively_optimal_assignment(self):
        """z0 = [0, 10, 20], z1 = [21, 1, 11]. By inspection the optimal
        pairing is 0<->1, 10<->11, 20<->21 (each at squared-distance 1,
        totaling 3) — verified to be the global optimum since every term
        simultaneously achieves its row-minimum under a valid permutation
        (row0->col1, row1->col2, row2->col0, using each column once)."""
        z0 = torch.tensor([[[0.0]], [[10.0]], [[20.0]]])
        z1 = torch.tensor([[[21.0]], [[1.0]], [[11.0]]])

        order = compute_ot_coupling(z0, z1)
        assert order.tolist() == [1, 2, 0]

        paired = z1[order].squeeze()
        assert torch.allclose(paired, torch.tensor([1.0, 11.0, 21.0]))


class TestPaddingAwareness:
    def test_masked_coupling_differs_from_unmasked_when_padding_noise_biases_it(self):
        """Hand-constructed case where the pairing that's optimal on REAL
        content only (masked) is different from the pairing that looks
        optimal once padded-region noise is included (unmasked) — the
        exact bug this module's masking exists to avoid.

        Sample 0 represents a short protein (only position 0 is real;
        positions 1-2 are padding). Its z0 counterpart has large random
        noise at those padded positions (999) — realistic, since z0 is
        drawn i.i.d. over the full padded tensor regardless of the real
        protein's length. Sample 1 represents a full-length protein (no
        padding).

        On real content alone, z0[0] (10.2) is obviously closest to z1[0]
        (10.0), and z0[1] should pair with z1[1] — the identity pairing.
        But unmasked, z0[0]'s huge noise at its padded positions (999 vs
        z1's padding-zeros) swamps the cost matrix and makes the swapped
        pairing look cheaper instead."""
        z0 = torch.tensor([
            [[10.2], [999.0], [999.0]],
            [[50.0], [5.1],   [5.1]],
        ])
        z1 = torch.tensor([
            [[10.0], [0.0], [0.0]],
            [[10.5], [5.0], [5.0]],
        ])
        padding_mask = torch.tensor([
            [False, True,  True],
            [False, False, False],
        ])

        order_unmasked = compute_ot_coupling(z0, z1, padding_mask=None)
        order_masked = compute_ot_coupling(z0, z1, padding_mask=padding_mask)

        assert order_unmasked.tolist() == [1, 0], (
            "sanity check on the constructed example: padded noise should "
            "make the swap look cheaper when unmasked"
        )
        assert order_masked.tolist() == [0, 1], (
            "masked coupling must recover the identity pairing, which is "
            "correct on real content"
        )

    def test_fully_padded_column_does_not_crash(self):
        """n_valid.clamp(min=1) guard: a pair with zero mutually-valid
        positions must not divide by zero."""
        z0 = torch.randn(2, 4, 3)
        z1 = torch.randn(2, 4, 3)
        padding_mask = torch.tensor([
            [False, False, True, True],
            [True, True, True, True],  # fully padded — degenerate but must not crash
        ])
        order = compute_ot_coupling(z0, z1, padding_mask=padding_mask)
        assert sorted(order.tolist()) == [0, 1]
