import numpy as np
import pytest
from par_protein.exposure_bias import NoisyContextLearning, ScheduledSampling


def test_noisy_context_learning():
    noisy_ctx = NoisyContextLearning(base_noise_std=0.2)
    embeds = np.zeros((10, 32))

    noisy0 = noisy_ctx.inject_context_noise(embeds, scale_idx=0, seed=1)
    noisy1 = noisy_ctx.inject_context_noise(embeds, scale_idx=1, seed=1)

    assert noisy0.shape == (10, 32)
    assert not np.array_equal(noisy0, embeds)
    assert np.std(noisy0) > np.std(noisy1)


def test_scheduled_sampling_linear():
    sch = ScheduledSampling(strategy="linear", start_prob=1.0, end_prob=0.0, total_steps=100)
    assert sch.get_teacher_forcing_probability(0) == 1.0
    assert sch.get_teacher_forcing_probability(50) == 0.5
    assert sch.get_teacher_forcing_probability(100) == 0.0


def test_scheduled_sampling_strategies():
    for strat in ("exponential", "inverse_sigmoid"):
        sch = ScheduledSampling(strategy=strat, start_prob=1.0, end_prob=0.1, total_steps=100)
        p0 = sch.get_teacher_forcing_probability(0)
        p50 = sch.get_teacher_forcing_probability(50)
        p100 = sch.get_teacher_forcing_probability(100)
        assert p0 == 1.0
        assert p0 >= p50 >= p100
        assert p100 == 0.1


def test_invalid_strategy():
    sch = ScheduledSampling(strategy="invalid")
    with pytest.raises(ValueError):
        sch.get_teacher_forcing_probability(50)
