"""Motive: motion attribution mechanisms and ground-truth experiments."""

from .attribution import (
    compute_motion_mask,
    compute_motion_weighted_attribution,
    experiment_dynamics_vs_magnitude,
    experiment_frame_length_bias,
    experiment_motion_mask_localization,
    make_moving_square_video,
    normalize_frame_length_bias,
    true_motion_patch_mask,
)

__all__ = [
    "compute_motion_mask",
    "compute_motion_weighted_attribution",
    "experiment_dynamics_vs_magnitude",
    "experiment_frame_length_bias",
    "experiment_motion_mask_localization",
    "make_moving_square_video",
    "normalize_frame_length_bias",
    "true_motion_patch_mask",
]
