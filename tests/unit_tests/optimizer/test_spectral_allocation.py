# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
from torch import nn
from torch.optim import Adam

from examples.run_simple_mcore_train_loop import build_optimizers
from megatron.core.optimizer.emerging_optimizers import (
    _EMERGING_OPTIMIZERS,
    HAVE_EMERGING_OPTIMIZERS,
)
from megatron.core.optimizer.spectral_allocation import (
    TensorParallelSpectralAwareMuon,
    two_level_spectral_update,
)

requires_emerging_optimizers = pytest.mark.skipif(
    not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package is not installed"
)


def _whitened(momentum: torch.Tensor) -> torch.Tensor:
    """Exact orthogonalization U V^T of a full-rank matrix, for test oracles."""
    u, _, vh = torch.linalg.svd(momentum.float(), full_matrices=False)
    return u @ vh


class TestTwoLevelSpectralUpdate:
    def test_bulk_scale_one_is_identity(self):
        torch.manual_seed(0)
        momentum = torch.randn(16, 8)
        orth = _whitened(momentum)
        update = two_level_spectral_update(orth, momentum, bulk_scale=1.0)
        assert torch.equal(update, orth)

    def test_head_held_at_muon_scale_bulk_amplified(self):
        torch.manual_seed(0)
        m, n = 16, 8
        bulk_scale = 2.0
        # Anisotropic momentum: a dominant head direction over a small bulk.
        u1 = torch.randn(m, 1)
        v1 = torch.randn(1, n)
        momentum = 5.0 * (u1 @ v1) + 0.1 * torch.randn(m, n)
        orth = _whitened(momentum)

        update = two_level_spectral_update(
            orth, momentum, bulk_scale=bulk_scale, num_power_iters=30
        )

        # The head singular value stays at the Muon scale (1) while every bulk
        # singular value is amplified to bulk_scale.
        svals = torch.linalg.svdvals(update)
        expected = torch.cat(
            [torch.full((n - 1,), bulk_scale), torch.ones(1)]
        )
        assert torch.allclose(svals, expected, atol=2e-2)

    def test_zero_momentum_falls_back_to_plain_update(self):
        orth = _whitened(torch.randn(8, 8))
        update = two_level_spectral_update(orth, torch.zeros(8, 8), bulk_scale=1.5)
        assert torch.equal(update, orth)

    def test_dtype_preserved(self):
        torch.manual_seed(0)
        momentum = torch.randn(8, 8, dtype=torch.bfloat16)
        orth = _whitened(momentum).to(torch.bfloat16)
        update = two_level_spectral_update(orth, momentum, bulk_scale=1.5)
        assert update.dtype == torch.bfloat16


class TestExampleWiring:
    """Exercise the examples/run_simple_mcore_train_loop.py call site."""

    def _tiny_model(self):
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))

    def test_adam_default_single_optimizer(self):
        model = self._tiny_model()
        optimizers = build_optimizers(model, "adam")
        assert len(optimizers) == 1
        assert isinstance(optimizers[0], Adam)
        num_params = sum(p.numel() for g in optimizers[0].param_groups for p in g["params"])
        assert num_params == sum(p.numel() for p in model.parameters())

    def test_unknown_optimizer_rejected(self):
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            build_optimizers(self._tiny_model(), "sgd")

    @requires_emerging_optimizers
    def test_spectral_aware_muon_param_split(self):
        model = self._tiny_model()
        optimizers = build_optimizers(model, "spectral_aware_muon")
        assert isinstance(optimizers[0], TensorParallelSpectralAwareMuon)
        # Matrix weights go to spectral-aware Muon; 1-D norm params go to Adam.
        muon_shapes = {tuple(p.shape) for g in optimizers[0].param_groups for p in g["params"]}
        assert muon_shapes == {(8, 8)}
        assert len(optimizers) == 2
        assert isinstance(optimizers[1], Adam)
        adam_numel = sum(p.numel() for g in optimizers[1].param_groups for p in g["params"])
        assert adam_numel == 8 + 8 + 8  # linear bias + layernorm weight + layernorm bias

    @requires_emerging_optimizers
    def test_invalid_bulk_scale_rejected(self):
        with pytest.raises(ValueError, match="bulk_scale"):
            TensorParallelSpectralAwareMuon([torch.nn.Parameter(torch.randn(4, 4))], bulk_scale=0.5)


@requires_emerging_optimizers
def test_spectral_aware_muon_registered():
    entry = _EMERGING_OPTIMIZERS['spectral_aware_muon']
    assert entry.optimizer_cls is TensorParallelSpectralAwareMuon
