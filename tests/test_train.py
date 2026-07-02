"""
tests/test_train.py

Unit tests for the standalone, distributed-free helpers in training/train.py:
get_param_groups, compute_group_grad_norms, warmup_lambda, update_ema.

Importing training.train doesn't require torch.distributed to be
initialized — only calling main() does, which these tests never do.
Run with:

    pytest tests/test_train.py -v
"""

from __future__ import annotations

import torch
import pytest

from training.train import (
    get_param_groups,
    compute_group_grad_norms,
    warmup_lambda,
    update_ema,
)
from training.dit_model import FluxionDiT
from training.flow_matching import FlowMatcher


B, L, D_S, D_P = 2, 6, 16, 8
N_HEADS, D_FF, D_FREQ = 2, 32, 16


def _tiny_model() -> FluxionDiT:
    return FluxionDiT(
        d_single=D_S, d_pair=D_P, d_model=D_S,
        n_layers=3, n_heads=N_HEADS, d_ff=D_FF, d_freq=D_FREQ,
    )


class TestGetParamGroups:
    def test_splits_into_two_groups(self):
        model = _tiny_model()
        groups = get_param_groups(model, weight_decay=0.01)
        assert len(groups) == 2
        assert groups[0]["weight_decay"] == 0.01
        assert groups[1]["weight_decay"] == 0.0

    def test_covers_every_trainable_param_exactly_once(self):
        model = _tiny_model()
        groups = get_param_groups(model, weight_decay=0.01)
        grouped_ids = {id(p) for g in groups for p in g["params"]}
        model_ids = {id(p) for p in model.parameters() if p.requires_grad}
        assert grouped_ids == model_ids

    def test_1d_params_excluded_from_decay(self):
        model = _tiny_model()
        groups = get_param_groups(model, weight_decay=0.01)
        no_decay_ids = {id(p) for p in groups[1]["params"]}
        for name, param in model.named_parameters():
            if param.ndim <= 1 or name.endswith(".bias"):
                assert id(param) in no_decay_ids, f"{name} should have no weight decay"

    def test_adaLN_mod_weight_excluded_from_decay(self):
        """AdaLN-zero modulation weights start at exactly zero and only grow
        as the model learns to use conditioning — weight decay must not
        fight that emergence."""
        model = _tiny_model()
        groups = get_param_groups(model, weight_decay=0.01)
        no_decay_ids = {id(p) for p in groups[1]["params"]}

        assert id(model.final_layer.adaLN_mod[-1].weight) in no_decay_ids
        for block in model.blocks:
            assert id(block.adaLN_mod[-1].weight) in no_decay_ids

    def test_regular_2d_weights_still_decayed(self):
        model = _tiny_model()
        groups = get_param_groups(model, weight_decay=0.01)
        decay_ids = {id(p) for p in groups[0]["params"]}
        assert id(model.input_proj.weight) in decay_ids
        assert id(model.final_layer.linear.weight) in decay_ids


class TestComputeGroupGradNorms:
    def _backward_pass(self, model: FluxionDiT) -> None:
        matcher = FlowMatcher()
        z0   = torch.randn(B, L, D_S)
        z1   = torch.randn(B, L, D_S)
        pair = torch.randn(B, L, L, D_P)
        loss = matcher.compute_loss(model, z0, z1, pair)
        loss.backward()

    def test_zero_grad_norms_before_backward(self):
        model = _tiny_model()
        norms = compute_group_grad_norms(model)
        assert norms["total_grad_norm"] == 0.0
        assert norms["gate_grad_norm"] == 0.0

    def test_nonzero_total_after_backward(self):
        torch.manual_seed(0)
        model = _tiny_model()
        self._backward_pass(model)
        norms = compute_group_grad_norms(model)
        assert norms["total_grad_norm"] > 0.0

    def test_gate_norm_never_exceeds_total(self):
        torch.manual_seed(0)
        model = _tiny_model()
        self._backward_pass(model)
        norms = compute_group_grad_norms(model)
        assert norms["gate_grad_norm"] <= norms["total_grad_norm"] + 1e-8

    def test_gate_norm_matches_manual_subset_computation(self):
        """gate_grad_norm must equal the L2 norm of exactly the adaLN_mod /
        final_layer.linear gradients, computed independently here."""
        torch.manual_seed(0)
        model = _tiny_model()
        self._backward_pass(model)
        norms = compute_group_grad_norms(model)

        manual_sq = 0.0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if "adaLN_mod" in name or "final_layer.linear" in name:
                manual_sq += param.grad.detach().double().pow(2).sum().item()
        expected = manual_sq ** 0.5

        assert norms["gate_grad_norm"] == pytest.approx(expected, rel=1e-9)


class TestWarmupLambda:
    def test_linear_ramp_during_warmup(self):
        assert warmup_lambda(0, 100) == 0.0
        assert warmup_lambda(50, 100) == pytest.approx(0.5)
        assert warmup_lambda(100, 100) == 1.0

    def test_constant_after_warmup(self):
        assert warmup_lambda(101, 100) == 1.0
        assert warmup_lambda(10_000, 100) == 1.0

    def test_zero_warmup_steps_does_not_divide_by_zero(self):
        assert warmup_lambda(0, 0) == 1.0


class TestUpdateEma:
    def test_ema_moves_toward_model_weights(self):
        torch.manual_seed(0)
        model = _tiny_model()
        ema_model = _tiny_model()
        # Force a known gap between model and ema weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        before = {n: p.clone() for n, p in ema_model.named_parameters()}
        update_ema(ema_model, model, decay=0.9)
        for name, param in ema_model.named_parameters():
            model_param = dict(model.named_parameters())[name]
            expected = before[name] * 0.9 + model_param.detach() * 0.1
            assert torch.allclose(param, expected, atol=1e-6)

    def test_ema_unchanged_when_decay_is_one(self):
        torch.manual_seed(0)
        model = _tiny_model()
        ema_model = _tiny_model()
        before = {n: p.clone() for n, p in ema_model.named_parameters()}
        update_ema(ema_model, model, decay=1.0)
        for name, param in ema_model.named_parameters():
            assert torch.allclose(param, before[name])
