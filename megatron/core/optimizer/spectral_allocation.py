# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Spectral step-size allocation for orthogonalized (Muon) updates.

Implements the SAMuon-lite variant from "Spectral Allocation: Why Muon Outperforms
Adam, and How to Improve Muon" (https://arxiv.org/abs/2608.25990v1). The paper's
spectral probing shows the loss-optimal per-direction step size is anisotropic: the
top singular direction of the momentum (the volatile "head") tolerates only the Muon
step size, while the remaining bulk permits substantially larger steps. SAMuon-lite
approximates that profile with two levels: hold the head at the Muon scale and amplify
the bulk by a static factor, estimating the head direction with rank-one power
iteration on the momentum buffer. It adds no persistent optimizer state and negligible
FLOPs on top of Muon.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import log_single_rank

from .emerging_optimizers import (
    _EMERGING_OPTIMIZERS,
    HAVE_EMERGING_OPTIMIZERS,
    EmergingOptimizerEntry,
    TensorParallelMuon,
    _eopt_init_state_fn,
    _is_muon_excluded,
    _kwargs_from_config,
    _muon_config_to_kwargs,
)
from .optimizer_config import ParamKey, ParamPredicate

logger = logging.getLogger(__name__)


def _all_reduce_(tensor: torch.Tensor, group: Optional[torch.distributed.ProcessGroup]) -> None:
    """In-place all-reduce over ``group`` when distributed is initialized; no-op otherwise."""
    if (
        group is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.all_reduce(tensor, group=group)


def _rank_one_head(
    momentum: torch.Tensor,
    num_iters: int,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    partition_dim: Optional[int] = None,
    eps: float = 1e-12,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Estimate the top singular pair of ``momentum`` via power iteration.

    The returned pair ``(u, v)`` is sharded consistently with ``momentum`` so that
    ``torch.outer(u, v)`` is the local shard of the rank-one head matrix:

    - ``partition_dim == 0``: rows are sharded; ``u`` is the local row shard, ``v`` is full.
    - ``partition_dim == 1``: columns are sharded; ``u`` is full, ``v`` is the local shard.
    - ``partition_dim is None``: purely local computation (no collectives).

    Args:
        momentum: The (possibly TP-sharded) momentum buffer, 2-D.
        num_iters: Number of power iterations.
        tp_group: Process group the matrix is sharded over. Only used when
            ``partition_dim`` is not None.
        partition_dim: Dimension along which ``momentum`` is sharded, or None.
        eps: Norm floor below which the momentum is treated as zero.

    Returns:
        The ``(u, v)`` pair described above, or None if the momentum is (near-)zero,
        in which case there is no head direction to defend.
    """
    n = momentum.shape[1]
    work = momentum.float()
    if partition_dim == 1:
        # Local column shard of the full right vector.
        v = torch.ones(n, dtype=work.dtype, device=work.device)
    else:
        v = torch.full((n,), n**-0.5, dtype=work.dtype, device=work.device)

    for _ in range(num_iters):
        if partition_dim == 0:
            u = work @ v  # local row shard of M v
            u_norm_sq = u.dot(u)
            _all_reduce_(u_norm_sq, tp_group)
            if u_norm_sq <= eps:
                return None
            u = u / u_norm_sq.sqrt()
            v = work.T @ u  # partial sum over row shards
            _all_reduce_(v, tp_group)
        elif partition_dim == 1:
            u = work @ v  # partial sum over column shards
            _all_reduce_(u, tp_group)
        else:
            u = work @ v

        if partition_dim != 0:
            u_norm = u.norm()
            if u_norm <= eps:
                return None
            u = u / u_norm
            v = work.T @ u

        if partition_dim == 1:
            # v is the local column shard; the full norm needs a reduction.
            v_norm_sq = v.dot(v)
            _all_reduce_(v_norm_sq, tp_group)
            if v_norm_sq <= eps:
                return None
            v = v / v_norm_sq.sqrt()
        else:
            v_norm = v.norm()
            if v_norm <= eps:
                return None
            v = v / v_norm

    return u, v


def two_level_spectral_update(
    orth_update: torch.Tensor,
    momentum: torch.Tensor,
    bulk_scale: float,
    num_power_iters: int = 1,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    partition_dim: Optional[int] = None,
) -> torch.Tensor:
    """Reweight an orthogonalized update with a two-level spectral prior.

    Holds the momentum's top singular direction (the head) at the Muon scale and
    amplifies all remaining (bulk) directions by ``bulk_scale``:

        update = bulk_scale * orth - (bulk_scale - 1) * (u^T orth v) * u v^T

    Args:
        orth_update: The orthogonalized, Muon-scaled update (possibly TP-sharded).
        momentum: The momentum buffer the update was orthogonalized from, sharded
            identically to ``orth_update``.
        bulk_scale: Amplification factor for the bulk directions. 1.0 recovers the
            vanilla Muon update exactly.
        num_power_iters: Power iterations used to estimate the head direction.
        tp_group: Process group the tensors are sharded over (for the collectives
            that make the head estimate TP-invariant). Only used when
            ``partition_dim`` is not None.
        partition_dim: Dimension along which the tensors are sharded, or None.

    Returns:
        The reweighted update, same shape, dtype, and sharding as ``orth_update``.
    """
    if bulk_scale == 1.0:
        return orth_update

    head = _rank_one_head(momentum, num_power_iters, tp_group, partition_dim)
    if head is None:
        return orth_update
    u, v = head

    # u^T orth v, reduced across shards so every rank scales its head shard equally.
    coeff = (u * (orth_update.float() @ v)).sum()
    if partition_dim is not None:
        _all_reduce_(coeff, tp_group)
    update = bulk_scale * orth_update - (bulk_scale - 1.0) * coeff.to(
        orth_update.dtype
    ) * torch.outer(u, v).to(orth_update.dtype)
    return update.to(orth_update.dtype)


class TensorParallelSpectralAwareMuon(TensorParallelMuon):
    """Tensor Parallel spectral-aware Muon (SAMuon-lite).

    Extends :class:`TensorParallelMuon` by reweighting each orthogonalized update with
    the two-level spectral prior of :func:`two_level_spectral_update`: the momentum's
    top singular direction stays at the Muon scale while the bulk is amplified by
    ``bulk_scale``. With ``bulk_scale=1.0`` this reduces exactly to Muon.

    Args:
        bulk_scale: Amplification factor for the bulk singular directions. The paper
            measures values in roughly the 1.2-2.0 range across model scales; 1.0
            recovers vanilla Muon.
        head_power_iters: Power iterations for the rank-one head estimate. One
            iteration already captures most of the gain and keeps overhead near zero.

    All other arguments are forwarded to :class:`TensorParallelMuon`.
    """

    def __init__(
        self,
        *args: Any,
        bulk_scale: float = 1.0,
        head_power_iters: int = 1,
        **kwargs: Any,
    ) -> None:
        if bulk_scale < 1.0:
            raise ValueError(f"bulk_scale must be at least 1.0, got {bulk_scale}")
        if head_power_iters < 1:
            raise ValueError(f"head_power_iters must be at least 1, got {head_power_iters}")
        super().__init__(*args, **kwargs)
        self.bulk_scale = bulk_scale
        self.head_power_iters = head_power_iters

        base_scaled_orthogonalize_fn = self.scaled_orthogonalize_fn

        def spectral_aware_scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
            tp_mode_this_group: str = self.tp_mode,
        ) -> torch.Tensor:
            orth_grad = base_scaled_orthogonalize_fn(
                grad, tp_group, partition_dim, tp_mode_this_group
            )
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Applying two-level spectral prior, bulk_scale={self.bulk_scale}, '
                f'head_power_iters={self.head_power_iters}',
            )
            return two_level_spectral_update(
                orth_grad,
                grad,
                self.bulk_scale,
                self.head_power_iters,
                tp_group if partition_dim is not None else None,
                partition_dim,
            )

        self.scaled_orthogonalize_fn = spectral_aware_scaled_orthogonalize_fn


def _spectral_aware_muon_config_to_kwargs(
    config, model_chunks, pg_collection: Optional[ProcessGroupCollection]
) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelSpectralAwareMuon constructor kwargs."""
    kwargs = _muon_config_to_kwargs(config, model_chunks, pg_collection)
    kwargs.update(_kwargs_from_config(TensorParallelSpectralAwareMuon, "muon", config))
    return kwargs


def register_spectral_aware_muon() -> None:
    """Register 'spectral_aware_muon' in the emerging optimizer registry."""
    _EMERGING_OPTIMIZERS.setdefault(
        'spectral_aware_muon',
        EmergingOptimizerEntry(
            optimizer_cls=TensorParallelSpectralAwareMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_spectral_aware_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(predicate=ParamPredicate(name="muon_excluded", fn=_is_muon_excluded)): {
                    'optimizer': 'adam'
                }
            },
        ),
    )


if HAVE_EMERGING_OPTIMIZERS:
    register_spectral_aware_muon()
