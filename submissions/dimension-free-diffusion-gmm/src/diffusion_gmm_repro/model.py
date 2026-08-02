"""Analytic operations for finite mixtures of isotropic Gaussians."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .schedule import DDPMSchedule


@dataclass(frozen=True)
class IsotropicGMM:
    """A normalized finite mixture with one scalar variance per component."""

    weights: NDArray[np.float64]
    means: NDArray[np.float64]
    variances: NDArray[np.float64]

    def __init__(
        self, weights: ArrayLike, means: ArrayLike, variances: ArrayLike
    ) -> None:
        normalized_weights = np.asarray(weights, dtype=np.float64)
        component_means = np.asarray(means, dtype=np.float64)
        component_variances = np.asarray(variances, dtype=np.float64)

        if component_means.ndim != 2:
            raise ValueError("means must be a two-dimensional array")
        if normalized_weights.ndim != 1 or component_variances.ndim != 1:
            raise ValueError("weights and variances must be one-dimensional")
        if not (
            len(normalized_weights)
            == len(component_means)
            == len(component_variances)
        ):
            raise ValueError("component counts must agree")
        if len(normalized_weights) == 0:
            raise ValueError("the mixture must contain at least one component")
        if not np.all(np.isfinite(component_means)):
            raise ValueError("means must be finite")
        if not np.all(np.isfinite(normalized_weights)) or np.any(
            normalized_weights <= 0.0
        ):
            raise ValueError("weights must be finite and positive")
        if not np.all(np.isfinite(component_variances)) or np.any(
            component_variances <= 0.0
        ):
            raise ValueError("variances must be finite and positive")

        object.__setattr__(
            self, "weights", normalized_weights / normalized_weights.sum()
        )
        object.__setattr__(self, "means", component_means)
        object.__setattr__(self, "variances", component_variances)

    @property
    def dimension(self) -> int:
        return self.means.shape[1]

    def _points(self, x: ArrayLike) -> NDArray[np.float64]:
        points = np.asarray(x, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError(f"x must have shape (n, {self.dimension})")
        return points

    def _component_log_density(
        self, points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        differences = points[:, None, :] - self.means[None, :, :]
        squared_distance = np.sum(differences * differences, axis=-1)
        normalizers = self.dimension * np.log(2.0 * np.pi * self.variances)
        return (
            np.log(self.weights)[None, :]
            - 0.5 * normalizers[None, :]
            - 0.5 * squared_distance / self.variances[None, :]
        )

    def _responsibilities(
        self, points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        component_logs = self._component_log_density(points)
        maxima = np.max(component_logs, axis=1, keepdims=True)
        unnormalized = np.exp(component_logs - maxima)
        return unnormalized / np.sum(unnormalized, axis=1, keepdims=True)

    def log_density(self, x: ArrayLike) -> NDArray[np.float64]:
        points = self._points(x)
        component_logs = self._component_log_density(points)
        maxima = np.max(component_logs, axis=1)
        return maxima + np.log(
            np.sum(np.exp(component_logs - maxima[:, None]), axis=1)
        )

    def score(self, x: ArrayLike) -> NDArray[np.float64]:
        points = self._points(x)
        responsibilities = self._responsibilities(points)
        component_scores = (self.means[None, :, :] - points[:, None, :]) / (
            self.variances[None, :, None]
        )
        return np.sum(responsibilities[:, :, None] * component_scores, axis=1)

    def score_jacobian_trace(self, x: ArrayLike) -> NDArray[np.float64]:
        points = self._points(x)
        responsibilities = self._responsibilities(points)
        component_scores = (self.means[None, :, :] - points[:, None, :]) / (
            self.variances[None, :, None]
        )
        component_terms = np.sum(component_scores**2, axis=-1) - (
            self.dimension / self.variances[None, :]
        )
        mixture_score = np.sum(
            responsibilities[:, :, None] * component_scores, axis=1
        )
        return np.sum(responsibilities * component_terms, axis=1) - np.sum(
            mixture_score**2, axis=1
        )


def forward_gmm(
    target: IsotropicGMM, *, schedule: DDPMSchedule, time: int
) -> IsotropicGMM:
    """Return the forward noised mixture at one schedule index."""
    if time < 0 or time > schedule.steps:
        raise ValueError("time must be between 0 and schedule.steps")
    alpha_bar_t = schedule.alpha_bar[time]
    return IsotropicGMM(
        target.weights,
        np.sqrt(alpha_bar_t) * target.means,
        alpha_bar_t * target.variances + (1.0 - alpha_bar_t),
    )


def score_at_time(
    target: IsotropicGMM,
    x: ArrayLike,
    *,
    schedule: DDPMSchedule,
    time: int,
) -> NDArray[np.float64]:
    """Evaluate the exact score of the forward noised target at ``time``."""
    return forward_gmm(target, schedule=schedule, time=time).score(x)
