import numpy as np
import pytest

from diffusion_gmm_repro.schedule import paper_schedule


def test_paper_schedule_satisfies_equation_14() -> None:
    schedule = paper_schedule(steps=128, c0=2.0, c1=10.0)
    assert schedule.alpha_bar[0] == 1.0
    assert schedule.alpha_bar[128] == pytest.approx(128.0**-2.0)
    for t in range(128, 1, -1):
        expected = schedule.alpha_bar[t] + (
            10.0 * np.log(128.0) / 128.0
        ) * schedule.alpha_bar[t] * (1.0 - schedule.alpha_bar[t])
        assert schedule.alpha_bar[t - 1] == pytest.approx(expected)
    np.testing.assert_allclose(
        schedule.alpha[1:],
        schedule.alpha_bar[1:] / schedule.alpha_bar[:-1],
    )


@pytest.mark.parametrize(
    ("steps", "c0", "c1", "message"),
    [
        (1, 2.0, 10.0, "steps"),
        (128, 0.0, 10.0, "positive"),
        (128, 2.0, 8.0, "greater than 4"),
    ],
)
def test_paper_schedule_rejects_out_of_scope_parameters(
    steps: int, c0: float, c1: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        paper_schedule(steps=steps, c0=c0, c1=c1)
