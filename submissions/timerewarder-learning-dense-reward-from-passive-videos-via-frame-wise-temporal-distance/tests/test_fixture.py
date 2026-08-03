import json

from timerewarder_repro.fixture import run_fixture


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def test_fixture_specification_is_fixed_and_action_free() -> None:
    result = run_fixture()

    assert result["specification"] == {
        "seed": 20260725,
        "dtype": "float64",
        "cpu_threads": 1,
        "trajectory_length": 9,
        "train_trajectories": 32,
        "test_trajectories": 8,
        "train_ordered_pairs": 2304,
        "test_ordered_pairs": 576,
        "iterations": 250,
        "learning_rate": 0.2,
        "initial_weights": "zeros",
        "data_order": "lexicographic",
    }
    assert "actions" not in json.dumps(result).lower()
    assert result["acceptance_threshold"] is None
    assert result["diagnostic_only"] is True


def test_fixture_measurements_are_byte_identical() -> None:
    first = canonical_json(run_fixture()["measurements"])
    second = canonical_json(run_fixture()["measurements"])

    assert first == second
    measurements = run_fixture()["measurements"]
    assert set(measurements) == {"positive", "permuted_label_control"}
    for measurement in measurements.values():
        assert set(measurement) == {
            "decoded_temporal_distance_mae",
            "nonzero_adjacent_reward_density",
            "forward_reverse_sign_accuracy",
            "train_ordered_pairs",
            "test_ordered_pairs",
        }
        assert measurement["train_ordered_pairs"] == 2304
        assert measurement["test_ordered_pairs"] == 576
