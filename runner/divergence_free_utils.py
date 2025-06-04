"""
Utility functions for divergence-free vector fields in protein design.

Adapted from image generation methods for protein rigid body transformations.
"""

import torch
import numpy as np
from typing import Dict, Any


def score_si_linear_protein(
    rigids_t, t_batch, rot_vectorfield, trans_vectorfield, eps=1e-8
):
    """
    Compute score function for protein rigid body transformations.

    Args:
        rigids_t: Current rigid body states [B, N, 7] (quaternion + translation)
        t_batch: Time values [B]
        rot_vectorfield: Rotation vector field [B, N, 3]
        trans_vectorfield: Translation vector field [B, N, 3]
        eps: Small epsilon for numerical stability

    Returns:
        Dictionary with rotation and translation scores
    """
    t_scalar = t_batch[0].item()

    if t_scalar < eps:
        # At t=0, score is approximately -x
        rot_score = -rigids_t[..., :4]  # Quaternion part
        trans_score = -rigids_t[..., 4:]  # Translation part
    else:
        # Reshape time for broadcasting
        t = t_batch.view(-1, *([1] * (rigids_t.ndim - 1)))
        one_minus_t = 1.0 - t

        # For rotations (quaternion part)
        rot_score = -((one_minus_t * rot_vectorfield + rigids_t[..., :4]) / t)

        # For translations
        trans_score = -((one_minus_t * trans_vectorfield + rigids_t[..., 4:]) / t)

    return {"rot_score": rot_score, "trans_score": trans_score}


def divfree_swirl_protein(
    rigids_t, t_batch, rot_vectorfield, trans_vectorfield, lambda_div=0.2, eps=1e-8
):
    """
    Generate divergence-free vector fields for protein rigid body transformations.

    Args:
        rigids_t: Current rigid body states [B, N, 7]
        t_batch: Time values [B]
        rot_vectorfield: Rotation vector field [B, N, 3]
        trans_vectorfield: Translation vector field [B, N, 3]
        lambda_div: Scale factor for divergence-free field
        eps: Small epsilon for numerical stability

    Returns:
        Dictionary with divergence-free rotation and translation fields
    """
    # Get scores
    scores = score_si_linear_protein(
        rigids_t, t_batch, rot_vectorfield, trans_vectorfield, eps
    )
    rot_score = scores["rot_score"]
    trans_score = scores["trans_score"]

    # Generate random noise with same shape as vector fields
    eps_rot = torch.randn_like(rot_vectorfield)
    eps_trans = torch.randn_like(trans_vectorfield)

    # Process rotation divergence-free field
    # Use only the first 3 components of quaternion for score computation
    rot_score_3d = rot_score[..., :3]  # [B, N, 3]

    # Compute projection for rotation
    dims_rot = tuple(range(2, rot_score_3d.ndim))  # Sum over spatial dimensions
    dot_rot = (eps_rot * rot_score_3d).sum(dim=dims_rot, keepdim=True)
    s_norm_rot = (
        torch.linalg.vector_norm(rot_score_3d, dim=dims_rot, keepdim=True) + eps
    )
    s_norm2_rot = s_norm_rot.pow(2)
    proj_rot = dot_rot / s_norm2_rot
    w_rot = eps_rot - proj_rot * rot_score_3d

    # Process translation divergence-free field
    dims_trans = tuple(range(2, trans_score.ndim))  # Sum over spatial dimensions
    dot_trans = (eps_trans * trans_score).sum(dim=dims_trans, keepdim=True)
    s_norm_trans = (
        torch.linalg.vector_norm(trans_score, dim=dims_trans, keepdim=True) + eps
    )
    s_norm2_trans = s_norm_trans.pow(2)
    proj_trans = dot_trans / s_norm2_trans
    w_trans = eps_trans - proj_trans * trans_score

    return {"rot_divfree": lambda_div * w_rot, "trans_divfree": lambda_div * w_trans}


def apply_divergence_free_step(
    rigids_t, rot_vectorfield, trans_vectorfield, t_batch, dt, lambda_div=0.2
):
    """
    Apply a single step with divergence-free vector fields.

    Args:
        rigids_t: Current rigid body states [B, N, 7]
        rot_vectorfield: Rotation vector field [B, N, 3]
        trans_vectorfield: Translation vector field [B, N, 3]
        t_batch: Time values [B]
        dt: Time step
        lambda_div: Scale factor for divergence-free field

    Returns:
        Updated rotation and translation vector fields with divergence-free components
    """
    # Get divergence-free fields
    divfree_fields = divfree_swirl_protein(
        rigids_t, t_batch, rot_vectorfield, trans_vectorfield, lambda_div
    )

    # Add divergence-free components to original vector fields
    enhanced_rot_vectorfield = rot_vectorfield + divfree_fields["rot_divfree"]
    enhanced_trans_vectorfield = trans_vectorfield + divfree_fields["trans_divfree"]

    return enhanced_rot_vectorfield, enhanced_trans_vectorfield
