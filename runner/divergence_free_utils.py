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
        # Reshape time for broadcasting with u_t shape
        one_minus_t = 1.0 - t_scalar
        t = t_scalar
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
    # Generate noise with same shape as velocity field
    eps_raw = torch.randn_like(u_t)
    score = score_si_linear(x, t_batch, u_t)

    # Compute projection to make noise divergence-free
    dims = tuple(range(1, u_t.ndim))
    dot = (eps_raw * score).sum(dim=dims, keepdim=True)
    s_norm = torch.linalg.vector_norm(score, dim=dims, keepdim=True) + eps
    s_norm2 = s_norm.pow(2)
    proj = dot / s_norm2
    w = eps_raw - proj * score

    return w


def make_divergence_free(noise, x, t_batch, u_t, eps=1e-8):
    """
    Project arbitrary noise to be divergence-free using score projection.
    This is equivalent to the make_divergence_free function from the image repo.

    Args:
        noise: Arbitrary noise tensor to be made divergence-free
        x: Current state
        t_batch: Time values [B]
        u_t: Velocity field
        eps: Small epsilon for numerical stability

    Returns:
        Divergence-free version of the input noise
    """
    score = score_si_linear(x, t_batch, u_t)

    # Project out component parallel to score
    dims = tuple(range(1, len(noise.shape)))  # All dims except batch
    dot = (noise * score).sum(dim=dims, keepdim=True)
    s_norm2 = torch.linalg.vector_norm(score, dim=dims, keepdim=True).pow(2) + eps
    proj = dot / s_norm2

    divergence_free_noise = noise - proj * score
    return divergence_free_noise


def calculate_euclidean_repulsion_forces(positions, alpha_t=1.0):
    """
    Calculate Euclidean repulsion forces between particles.
    Based on the particle_guidance_forces function from the image repo.

    Args:
        positions: Current positions [batch_size, ...]
        alpha_t: Guidance strength (time-dependent)

    Returns:
        Repulsion forces with same shape as positions
    """
    batch_size = positions.shape[0]
    if batch_size == 1:
        return torch.zeros_like(positions)

    # Flatten positions for distance calculation
    pos_flat = positions.flatten(1)  # [batch_size, flattened_dims]

    # Vectorized computation of pairwise differences
    diff = pos_flat.unsqueeze(1) - pos_flat.unsqueeze(
        0
    )  # [batch_size, batch_size, features]

    # Compute pairwise distances with small epsilon
    distances = (
        torch.norm(diff, dim=2, keepdim=True) + 1e-6
    )  # [batch_size, batch_size, 1]

    # Force proportional to 1/distance³
    forces = diff / (distances**3)  # [batch_size, batch_size, features]

    # Exclude diagonal elements (self-interaction)
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=positions.device).unsqueeze(
        2
    )
    forces = forces * mask

    # Sum forces for each particle
    forces_flat = forces.sum(dim=1)  # [batch_size, features]

    # Reshape back to original dimensions and apply guidance strength
    forces = forces_flat.view_as(positions)
    return alpha_t * forces


def divfree_max_noise(
    x,
    t_batch,
    u_t,
    lambda_div=0.2,
    repulsion_strength=0.02,
    noise_schedule_end_factor=0.7,
    min_t=0.01,
    eps=1e-8,
):
    """
    Generate divergence-free max noise with linear time schedule and particle repulsion.
    This combines the divergence-free noise generation with particle repulsion forces
    and time-dependent scaling, following the approach from the image repository.

    Args:
        x: Current state
        t_batch: Time values [B]
        u_t: Velocity field
        lambda_div: Scale factor for divergence-free field
        repulsion_strength: Strength of repulsion forces
        noise_schedule_end_factor: End factor for time-dependent noise scaling
        min_t: Minimum time value for normalization
        eps: Small epsilon for numerical stability

    Returns:
        Scaled divergence-free noise with particle repulsion
    """
    # Calculate time-dependent noise scaling (matching image repo approach)
    t_scalar = t_batch[0].item()
    normalized_time = (1.0 - t_scalar) / (1.0 - min_t)  # 0 at start, 1 at end
    noise_scale_factor = 1.0 + (noise_schedule_end_factor - 1.0) * normalized_time

    # Generate standard divergence-free noise
    w_divfree = divfree_swirl_si(x, t_batch, None, u_t, eps)

    # Calculate particle repulsion forces
    raw_repulsion_forces = calculate_euclidean_repulsion_forces(x)

    # Regularize repulsion relative to Gaussian noise magnitude (matching image repo)
    dims = tuple(range(1, w_divfree.ndim))
    gaussian_magnitude = torch.linalg.vector_norm(w_divfree, dim=dims).mean()
    repulsion_magnitude = torch.linalg.vector_norm(
        raw_repulsion_forces, dim=dims
    ).mean()

    if repulsion_magnitude > 1e-8:
        regularization_factor = gaussian_magnitude / repulsion_magnitude
        regularized_repulsion = raw_repulsion_forces * regularization_factor
    else:
        regularized_repulsion = raw_repulsion_forces

    # Scale repulsion by strength factor
    scaled_repulsion = regularized_repulsion * repulsion_strength

    # Make repulsion forces divergence-free
    repulsion_divfree = make_divergence_free(scaled_repulsion, x, t_batch, u_t, eps)

    # Combine divergence-free noise and repulsion
    total_divfree_noise = w_divfree + repulsion_divfree

    # Apply time-dependent scaling and lambda scaling
    return lambda_div * noise_scale_factor * total_divfree_noise
