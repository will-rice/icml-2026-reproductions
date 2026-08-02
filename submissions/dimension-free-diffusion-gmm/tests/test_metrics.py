"""Tests for calibrated discrepancy metrics."""

import numpy as np
import pytest
from diffusion_gmm_repro.metrics import (
    DEFAULT_SEED,
    _histogram_gaussian_convolution,
    bootstrap_interval,
    classifier_tv_lower_bound,
    linear_mmd,
    log_slope_interval,
    partition_tv_lower_bound,
    plugin_tv_diagnostic,
)
from diffusion_gmm_repro.model import IsotropicGMM

EXPECTED_RECORD_KEYS = {
    "metric_kind",
    "estimate",
    "lower_95",
    "upper_95",
    "calibration_floor",
    "samples",
    "seed",
}


def test_partition_tv_is_zero_for_identical_exact_bin_probabilities() -> None:
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    assert partition_tv_lower_bound(probabilities, probabilities) == 0.0


def test_partition_tv_matches_two_bin_example() -> None:
    assert partition_tv_lower_bound(
        np.array([0.75, 0.25]), np.array([0.25, 0.75])
    ) == pytest.approx(0.5)


def test_partition_tv_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        partition_tv_lower_bound([np.nan, 0.5], [0.5, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        partition_tv_lower_bound([-0.1, 1.1], [0.5, 0.5])
    with pytest.raises(ValueError, match="shape"):
        partition_tv_lower_bound([0.5, 0.5], [0.3, 0.3, 0.4])


def test_linear_mmd_separates_shifted_samples() -> None:
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(8192, 2))
    same = rng.normal(size=(8192, 2))
    shifted = rng.normal(loc=0.5, size=(8192, 2))
    assert linear_mmd(reference, shifted, bandwidth=1.0) > linear_mmd(
        reference, same, bandwidth=1.0
    )


def test_bootstrap_is_seed_deterministic() -> None:
    values = np.linspace(0.0, 1.0, 50)
    assert bootstrap_interval(values, seed=8, replicates=200) == bootstrap_interval(
        values, seed=8, replicates=200
    )
    assert bootstrap_interval(values, seed=None, replicates=200) == bootstrap_interval(
        values, seed=None, replicates=200
    )


def test_bootstrap_rejects_non_95_confidence_level() -> None:
    values = np.linspace(0.0, 1.0, 50)
    with pytest.raises(ValueError, match="0.95"):
        bootstrap_interval(values, confidence_level=0.90)


def test_metrics_reject_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        linear_mmd(np.array([[np.nan]]), np.array([[0.0]]), bandwidth=1.0)


def test_classifier_tv_is_calibrated_by_target_target_floor() -> None:
    rng = np.random.default_rng(11)
    first = rng.normal(size=(4096, 3))
    second = rng.normal(size=(4096, 3))
    record = classifier_tv_lower_bound(first, second, seed=12)
    assert record["estimate"] <= record["upper_95"]
    assert record["metric_kind"] == "classifier-induced-tv-lower-bound"
    assert "calibration_floor" in record
    assert record["samples"] == 4096


def test_classifier_calibration_reduces_target_separability_to_zero() -> None:
    rng = np.random.default_rng(20)
    target_sample1 = rng.normal(size=(2048, 2))
    target_sample2 = rng.normal(size=(2048, 2))
    shifted_sample = rng.normal(loc=1.5, size=(2048, 2))

    same_record = classifier_tv_lower_bound(
        target_sample1, target_sample2, seed=10, replicates=100
    )
    shifted_record = classifier_tv_lower_bound(
        target_sample1, shifted_sample, seed=10, replicates=100
    )

    assert same_record["estimate"] < 0.03
    assert shifted_record["estimate"] > 0.1


def test_plugin_tv_diagnostic_fixed_grid_histogram_discrete_convolution() -> None:
    target = IsotropicGMM(
        weights=[0.5, 0.5],
        means=[[-1.0], [1.0]],
        variances=[1.0, 1.0],
    )
    rng = np.random.default_rng(15)
    comp_idx = rng.choice(2, size=2048, p=[0.5, 0.5])
    samples = np.where(comp_idx == 0, -1.0, 1.0) + rng.normal(size=2048)

    record = plugin_tv_diagnostic(
        samples, target, grid_points=200, seed=42, replicates=100
    )
    assert record["metric_kind"] == "plugin-tv-diagnostic"
    assert record["lower_95"] <= record["estimate"] <= record["upper_95"]
    assert record["calibration_floor"] >= 0.0


def test_histogram_gaussian_convolution_helper_values_and_kde_divergence() -> None:
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
    dx = 1.0
    bandwidth = 1.0
    # Samples with bin counts [2, 1, 0] on bin_edges [0,1), [1,2), [2,3)
    samples = np.array([0.1, 0.1, 1.9])

    # Independently computed discrete kernel convolved with empirical density [2/3, 1/3, 0]
    radius = 4
    j = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * j**2) / np.sqrt(2.0 * np.pi)
    kernel_norm = kernel / np.sum(kernel)
    empirical = np.array([2.0 / 3.0, 1.0 / 3.0, 0.0])
    full_c = np.convolve(empirical, kernel_norm, mode="full")
    expected_density = full_c[4:7]

    actual_density = _histogram_gaussian_convolution(
        samples, bin_edges, dx, bandwidth
    )
    np.testing.assert_allclose(actual_density, expected_density)

    # Direct continuous sample-grid KDE on grid centers [0.5, 1.5, 2.5]
    grid_centers = np.array([0.5, 1.5, 2.5])
    diffs = (grid_centers[:, None] - samples[None, :]) / bandwidth
    direct_kde_density = np.mean(
        np.exp(-0.5 * diffs**2) / (np.sqrt(2.0 * np.pi) * bandwidth), axis=1
    )
    # Direct continuous sample-grid KDE differs significantly from discrete histogram convolution
    assert not np.allclose(direct_kde_density, expected_density)

    # Intra-bin sample shifts leave histogram discrete convolution strictly invariant
    shifted_samples = np.array([0.9, 0.9, 1.1])
    shifted_density = _histogram_gaussian_convolution(
        shifted_samples, bin_edges, dx, bandwidth
    )
    np.testing.assert_allclose(shifted_density, actual_density)


def test_plugin_tv_small_grid_kernel_overflow_regression() -> None:
    target = IsotropicGMM(weights=[1.0], means=[[0.0]], variances=[1.0])
    samples = np.array([-10.0, 10.0])
    record = plugin_tv_diagnostic(samples, target, grid_points=2, seed=42, replicates=10)
    assert record["samples"] == 2
    assert set(record.keys()) == EXPECTED_RECORD_KEYS


def test_plugin_tv_diagnostic_validates_grid_points() -> None:
    target = IsotropicGMM(weights=[1.0], means=[[0.0]], variances=[1.0])
    samples = np.array([0.0, 0.1, 0.2])
    with pytest.raises(ValueError, match="grid_points"):
        plugin_tv_diagnostic(samples, target, grid_points=1)


def test_default_seed_determinism_and_explicit_seed_reproducibility() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(512, 2))
    y = rng.normal(loc=0.5, size=(512, 2))
    target = IsotropicGMM(weights=[1.0], means=[[0.0, 0.0]], variances=[1.0])
    steps = np.array([32, 64, 128])
    metrics = np.array([0.5, 0.25, 0.125])

    mmd_default1 = linear_mmd(x, y, seed=None)
    mmd_default2 = linear_mmd(x, y)
    assert mmd_default1 == mmd_default2
    assert mmd_default1["seed"] == DEFAULT_SEED

    clf_default1 = classifier_tv_lower_bound(x, y, seed=None)
    clf_default2 = classifier_tv_lower_bound(x, y)
    assert clf_default1 == clf_default2
    assert clf_default1["seed"] == DEFAULT_SEED

    plugin_default1 = plugin_tv_diagnostic(x, target, seed=None, replicates=50)
    plugin_default2 = plugin_tv_diagnostic(x, target, replicates=50)
    assert plugin_default1 == plugin_default2
    assert plugin_default1["seed"] == DEFAULT_SEED

    slope_default1 = log_slope_interval(steps, metrics, seed=None)
    slope_default2 = log_slope_interval(steps, metrics)
    assert slope_default1 == slope_default2
    assert slope_default1["seed"] == DEFAULT_SEED

    clf_explicit1 = classifier_tv_lower_bound(x, y, seed=123)
    clf_explicit2 = classifier_tv_lower_bound(x, y, seed=123)
    assert clf_explicit1 == clf_explicit2
    assert clf_explicit1["seed"] == 123


def test_log_slope_interval_rejects_non_95_confidence_level() -> None:
    steps = np.array([32, 64, 128, 256])
    metrics = 1.0 / steps
    with pytest.raises(ValueError, match="0.95"):
        log_slope_interval(steps, metrics, confidence_level=0.90)


def test_log_slope_interval_recovers_known_exponent() -> None:
    steps = np.array([32, 64, 128, 256])
    metrics = 1.0 / steps
    record = log_slope_interval(steps, metrics, seed=7, replicates=200)
    assert record["metric_kind"] == "log-slope-interval"
    assert record["estimate"] == pytest.approx(-1.0, abs=1e-5)
    assert record["lower_95"] <= record["estimate"] <= record["upper_95"]


def test_seven_key_record_contract() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(256, 2))
    y = rng.normal(loc=0.5, size=(256, 2))
    target = IsotropicGMM(weights=[1.0], means=[[0.0, 0.0]], variances=[1.0])
    steps = np.array([32, 64, 128])
    metrics = np.array([0.5, 0.25, 0.125])

    mmd_rec = linear_mmd(x, y)
    clf_rec = classifier_tv_lower_bound(x, y)
    plugin_rec = plugin_tv_diagnostic(x, target, replicates=50)
    slope_rec = log_slope_interval(steps, metrics)

    for rec in (mmd_rec, clf_rec, plugin_rec, slope_rec):
        assert set(rec.keys()) == EXPECTED_RECORD_KEYS
