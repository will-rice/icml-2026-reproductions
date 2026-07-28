import math

from conditional_dpo_repro.soft_margin import run_soft_margin_lane


def test_scaled_softplus_converges_to_hinge():
    result = run_soft_margin_lane()
    errors = result["max_abs_error_by_beta"]
    assert list(errors) == ["1", "4", "16", "64", "256"]
    assert errors["256"] < errors["64"] < errors["16"] < errors["4"]
    assert errors["256"] <= math.log(2.0) / 256.0 + 1e-12


def test_negative_margin_examples_include_wrong_preference():
    examples = run_soft_margin_lane()["negative_target_examples"]
    assert examples
    assert any(
        item["delta_ref"] < item["delta"] < 0.0
        and item["target_margin"] == item["delta_ref"]
        for item in examples
    )
