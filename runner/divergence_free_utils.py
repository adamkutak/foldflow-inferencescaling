"""
Utility functions for divergence-free vector fields in protein design.

Simplified approach for adding divergence-free noise to velocity fields.
"""

import torch
import numpy as np
from typing import Dict, Any


def score_si_linear(x, t_batch, u_t):
    """
    Convert velocity field to score function.

    Args:
        x: Current state
        t_batch: Time values [B]
        u_t: Velocity field

    Returns:
        Score function
    """
    t_scalar = t_batch[0].item()

    if t_scalar == 0.0:
        # limit t → 0  gives score = −x
        return -x
    else:
        one_minus_t = 1.0 - t_batch.view(-1, *([1] * (x.ndim - 1)))
        t = t_batch.view(-1, *([1] * (x.ndim - 1)))
        return -((one_minus_t * u_t + x) / t)


def divfree_swirl_si(x, t_batch, y, u_t, eps=1e-8):
    """
    Generate divergence-free noise field.

    Args:
        x: Current state
        t_batch: Time values [B]
        y: Not used in this version (kept for compatibility)
        u_t: Velocity field
        eps: Small epsilon for numerical stability

    Returns:
        Divergence-free noise field
    """
    eps_raw = torch.randn_like(x)
    score = score_si_linear(x, t_batch, u_t)

    dims = tuple(range(1, x.ndim))
    dot = (eps_raw * score).sum(dim=dims, keepdim=True)
    s_norm = torch.linalg.vector_norm(score, dim=dims, keepdim=True) + eps
    s_norm2 = s_norm.pow(2)
    proj = dot / s_norm2
    w = eps_raw - proj * score

    return w
