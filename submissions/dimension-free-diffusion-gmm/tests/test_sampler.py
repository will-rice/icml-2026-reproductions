import numpy as np
import pytest

from diffusion_gmm_repro.model import IsotropicGMM
from diffusion_gmm_repro.sampler import (
    constant_rms_profile,
    ddpm_sample,
    orthonormal_embedding,
)
from diffusion_gmm_repro.schedule import paper_schedule


def test_equation_9_preserves_standard_gaussian_moments() -> None:
    target = IsotropicGMM([1.0], [[0.0, 0.0]], [1.0])
    samples = ddpm_sample(
        target,
        schedule=paper_schedule(steps=128),
        samples=65536,
        seed=17,
    )
    np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=0.02)
    np.testing.assert_allclose(np.var(samples, axis=0), 1.0, atol=0.025)


def test_score_profiles_have_exact_requested_rms() -> None:
    for shape in ("uniform", "front-loaded", "back-loaded"):
        profile = constant_rms_profile(steps=128, rms=0.04, shape=shape)
        assert np.sqrt(np.mean(profile[1:] ** 2)) == pytest.approx(0.04, abs=1e-12)


def test_random_blocks_are_invariant_to_batch_grouping() -> None:
    target = IsotropicGMM([0.5, 0.5], [[-1.0], [1.0]], [1.0, 1.0])
    kwargs = {
        "schedule": paper_schedule(steps=128),
        "samples": 3072,
        "seed": 23,
    }
    np.testing.assert_array_equal(
        ddpm_sample(target, blocks_per_batch=1, **kwargs),
        ddpm_sample(target, blocks_per_batch=3, **kwargs),
    )


def test_orthonormal_embedding_uses_only_active_coordinates() -> None:
    embedding = orthonormal_embedding(active_dimension=2, ambient_dimension=5)
    assert embedding.shape == (5, 2)
    np.testing.assert_allclose(embedding.T @ embedding, np.eye(2))
    np.testing.assert_allclose(embedding[2:], 0.0)


def test_sampler_rejects_nonunit_target_variances() -> None:
    target = IsotropicGMM([1.0], [[0.0]], [2.0])
    with pytest.raises(ValueError, match="unit"):
        ddpm_sample(target, schedule=paper_schedule(steps=128), samples=1, seed=1)
