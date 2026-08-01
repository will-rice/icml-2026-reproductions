"""Core attribution and motion masking algorithms for Motive, with
ground-truth experiments.

The mechanism functions operate on real tensors. The experiment functions
build deterministic synthetic videos with known moving regions and measure
the mechanisms against that ground truth; every reported number is computed.
Benchmark fine-tuning results (VBench) and the paper's human study are out of
reach of this package and are reported as unreplicated by the evidence
generator — never synthesized here.
"""

import numpy as np
import torch
import torch.nn.functional as F

PATCH = 16
FRAME = 64


def compute_motion_mask(frames: torch.Tensor, patch_size: int = PATCH, threshold: float = 0.05) -> torch.Tensor:
    """Detect motion between consecutive frames and compute motion-magnitude patch masks.

    Args:
        frames: Video frames tensor of shape (B, T, C, H, W).
        patch_size: Size of spatial patches.
        threshold: Motion detection threshold.

    Returns:
        Motion mask of shape (B, T-1, H // patch_size, W // patch_size).
    """
    if frames.dim() == 4:
        frames = frames.unsqueeze(0)
    B, T, C, H, W = frames.shape
    if T < 2:
        return torch.ones((B, 1, H // patch_size, W // patch_size), dtype=torch.float32)

    diff = torch.abs(frames[:, 1:] - frames[:, :-1])
    diff_mean = diff.mean(dim=2, keepdim=True)

    ph, pw = H // patch_size, W // patch_size
    diff_patches = diff_mean.view(B, T - 1, 1, ph, patch_size, pw, patch_size)
    patch_motion = diff_patches.mean(dim=(4, 6)).squeeze(2)

    mask = torch.where(patch_motion > threshold, patch_motion / (patch_motion.max() + 1e-8), patch_motion * 0.1)
    return mask


def compute_motion_weighted_attribution(
    model_grad: torch.Tensor,
    motion_mask: torch.Tensor,
    patch_size: int = PATCH,
) -> float:
    """Compute motion-weighted loss mask attribution gradient norm."""
    B, T, C, H, W = model_grad.shape
    ph, pw = H // patch_size, W // patch_size

    mask_expanded = F.interpolate(
        motion_mask.view(B * (T - 1), 1, ph, pw),
        size=(H, W),
        mode="nearest",
    ).view(B, T - 1, 1, H, W)
    mask_full = torch.cat([mask_expanded[:, :1], mask_expanded], dim=1)

    weighted_grad = model_grad * mask_full
    return float(torch.norm(weighted_grad).item())


def normalize_frame_length_bias(raw_attributions: list, frame_lengths: list) -> list:
    """Frame-length bias normalization: S_norm = S_raw / sqrt(T_clip / T_ref)."""
    ref_length = float(np.mean(frame_lengths))
    normalized_scores = []
    for score, length in zip(raw_attributions, frame_lengths):
        norm_factor = np.sqrt(length / ref_length)
        normalized_scores.append(score / max(norm_factor, 1e-6))
    return normalized_scores


def make_moving_square_video(
    frames: int, square: int = 16, step: int = 4, seed: int = 5, jitter: bool = False
) -> torch.Tensor:
    """Deterministic synthetic video: static textured background plus one
    moving square (or, with jitter=True, per-frame incoherent noise motion)."""
    generator = torch.Generator().manual_seed(seed)
    background = torch.rand((3, FRAME, FRAME), generator=generator)
    video = torch.zeros((1, frames, 3, FRAME, FRAME))
    for t in range(frames):
        frame = background.clone()
        if jitter:
            noise = torch.rand((3, FRAME, FRAME), generator=generator)
            frame = 0.5 * frame + 0.5 * noise
        else:
            x = (step * t) % (FRAME - square)
            frame[:, 24 : 24 + square, x : x + square] = torch.rand(
                (3, square, square), generator=generator
            )
        video[0, t] = frame
    return video


def true_motion_patch_mask(frames: int, square: int = 16, step: int = 4) -> torch.Tensor:
    """Ground-truth patch mask: 1 where the moving square touches a patch."""
    ph = FRAME // PATCH
    truth = torch.zeros((frames - 1, ph, ph))
    for t in range(frames - 1):
        rows = range(24 // PATCH, (24 + square - 1) // PATCH + 1)
        x0 = (step * t) % (FRAME - square)
        x1 = (step * (t + 1)) % (FRAME - square)
        columns = set()
        for x in (x0, x1):
            columns.update(range(x // PATCH, (x + square - 1) // PATCH + 1))
        for r in rows:
            for c in columns:
                truth[t, r, c] = 1.0
    return truth


def experiment_motion_mask_localization(frames: int = 8) -> dict:
    """Measure how much of the mask's weight lands on truly-moving patches."""
    video = make_moving_square_video(frames)
    mask = compute_motion_mask(video)[0]
    truth = true_motion_patch_mask(frames)
    inside = float(mask[truth > 0].mean().item())
    outside = float(mask[truth == 0].mean().item())

    uniform_grad = torch.ones((1, frames, 3, FRAME, FRAME))
    masked_score = compute_motion_weighted_attribution(uniform_grad, mask.unsqueeze(0))
    unmasked_score = float(torch.norm(uniform_grad).item())
    mask_full = F.interpolate(
        mask.view(frames - 1, 1, FRAME // PATCH, FRAME // PATCH), size=(FRAME, FRAME), mode="nearest"
    )
    truth_full = F.interpolate(
        truth.view(frames - 1, 1, FRAME // PATCH, FRAME // PATCH), size=(FRAME, FRAME), mode="nearest"
    )
    dynamic_fraction = float(
        (mask_full * truth_full).sum().item() / (mask_full.sum().item() + 1e-8)
    )
    return {
        "mean_mask_in_moving_patches": round(inside, 4),
        "mean_mask_in_static_patches": round(outside, 4),
        "masked_attribution_norm": round(masked_score, 4),
        "unmasked_attribution_norm": round(unmasked_score, 4),
        "fraction_of_mask_weight_on_true_motion": round(dynamic_fraction, 4),
    }


def experiment_frame_length_bias() -> dict:
    """Measure raw attribution growth with clip length and its normalization.

    Clips share identical per-frame statistics; a genuinely more influential
    short clip (stronger injected motion signal) is out-ranked by a long clip
    under raw scores and recovered by normalization.
    """
    lengths = [8, 16, 32]
    raw_scores = []
    for i, length in enumerate(lengths):
        video = make_moving_square_video(length, seed=5 + i)
        mask = compute_motion_mask(video)
        generator = torch.Generator().manual_seed(17 + i)
        grads = torch.randn((1, length, 3, FRAME, FRAME), generator=generator)
        raw_scores.append(round(compute_motion_weighted_attribution(grads, mask), 4))
    normalized = [round(v, 4) for v in normalize_frame_length_bias(raw_scores, lengths)]

    strong_short = make_moving_square_video(8, step=12, seed=41)
    weak_long = make_moving_square_video(32, step=1, seed=42)
    scores = []
    for j, video in enumerate((strong_short, weak_long)):
        mask = compute_motion_mask(video)
        generator = torch.Generator().manual_seed(29 + j)
        grads = torch.randn(video.shape, generator=generator)
        scores.append(compute_motion_weighted_attribution(grads, mask))
    ranked_raw = ["strong_short", "weak_long"] if scores[0] > scores[1] else ["weak_long", "strong_short"]
    norm_pair = normalize_frame_length_bias(scores, [8, 32])
    ranked_norm = (
        ["strong_short", "weak_long"] if norm_pair[0] > norm_pair[1] else ["weak_long", "strong_short"]
    )
    return {
        "clip_lengths": lengths,
        "raw_scores": raw_scores,
        "normalized_scores": normalized,
        "raw_growth_ratio_longest_vs_shortest": round(raw_scores[-1] / raw_scores[0], 2),
        "normalized_growth_ratio_longest_vs_shortest": round(normalized[-1] / normalized[0], 2),
        "pair_raw_scores": [round(s, 4) for s in scores],
        "pair_normalized_scores": [round(s, 4) for s in norm_pair],
        "raw_ranking": ranked_raw,
        "normalized_ranking": ranked_norm,
    }


def experiment_dynamics_vs_magnitude(frames: int = 8) -> dict:
    """Show masked influence tracks target-like dynamics, not raw magnitude.

    A coherent translating square (matching the target clip's dynamic) is
    compared with an incoherent high-magnitude jitter clip. Influence proxy:
    cosine similarity between motion-masked temporal-difference fields.
    """
    target = make_moving_square_video(frames, seed=61)
    coherent = make_moving_square_video(frames, seed=62)
    jitter = make_moving_square_video(frames, seed=63, jitter=True)

    def masked_motion_field(video: torch.Tensor) -> torch.Tensor:
        mask = compute_motion_mask(video)
        diff = (video[:, 1:] - video[:, :-1]).mean(dim=2)
        mask_full = F.interpolate(
            mask.view(frames - 1, 1, FRAME // PATCH, FRAME // PATCH),
            size=(FRAME, FRAME),
            mode="nearest",
        ).view(1, frames - 1, FRAME, FRAME)
        return (diff.abs() * mask_full).flatten()

    target_field = masked_motion_field(target)
    coherent_cos = float(F.cosine_similarity(target_field, masked_motion_field(coherent), dim=0).item())
    jitter_cos = float(F.cosine_similarity(target_field, masked_motion_field(jitter), dim=0).item())
    coherent_energy = float((coherent[:, 1:] - coherent[:, :-1]).abs().mean().item())
    jitter_energy = float((jitter[:, 1:] - jitter[:, :-1]).abs().mean().item())
    return {
        "coherent_clip_motion_energy": round(coherent_energy, 4),
        "jitter_clip_motion_energy": round(jitter_energy, 4),
        "jitter_has_higher_raw_motion": bool(jitter_energy > coherent_energy),
        "coherent_influence_cosine": round(coherent_cos, 4),
        "jitter_influence_cosine": round(jitter_cos, 4),
        "influence_tracks_dynamics_not_magnitude": bool(coherent_cos > jitter_cos),
    }
