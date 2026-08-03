import pytest

from timerewarder_repro.checkpoint import anchor_indices, build_protocol

TASKS = {
    "basketball-v3": "basketball_20bins.pth",
    "button-press-topdown-v2": "button_press_topdown_20bins.pth",
    "disassemble-v2": "disassemble_20bins.pth",
    "door-open-v2": "door_open_20bins.pth",
    "drawer-open-v2": "drawer_open_20bins.pth",
    "lever-pull-v2": "lever_pull_20bins.pth",
    "plate-slide-v2": "plate_slide_20bins.pth",
    "stick-push-v2": "stick_push_20bins.pth",
    "window-close-v2": "window_close_20bins.pth",
    "window-open-v2": "window_open_20bins.pth",
}


@pytest.fixture
def dataset_manifest() -> dict[str, object]:
    return {
        "tasks": {
            task: {
                "held_out": [f"{task}/{ordinal}.mp4" for ordinal in range(1, 101)],
                "population_paths": [
                    f"{task}/population/{ordinal}.mp4" for ordinal in range(1, 4)
                ],
            }
            for task in TASKS
        }
    }


@pytest.fixture
def frame_counts(dataset_manifest: dict[str, object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task, annotations in dataset_manifest["tasks"].items():
        for ordinal, path in enumerate(annotations["held_out"], start=1):
            counts[path] = 100 + ordinal
        for ordinal, path in enumerate(annotations["population_paths"], start=1):
            counts[path] = 200 + ordinal
    return counts


def test_protocol_fixes_all_strata_ordinals_and_pair_counts(
    dataset_manifest: dict[str, object], frame_counts: dict[str, int]
) -> None:
    protocol = build_protocol(dataset_manifest, frame_counts)

    assert [item["task"] for item in protocol] == list(TASKS)
    assert {item["checkpoint"] for item in protocol} == set(TASKS.values())
    assert {tuple(item["held_out_ordinals"]) for item in protocol} == {
        (1, 26, 51, 76, 100)
    }
    assert (
        sum(
            len(video["ordered_pairs"]) for task in protocol for video in task["videos"]
        )
        == 1000
    )
    assert sum(len(task["videos"]) for task in protocol) == 50


@pytest.mark.parametrize(
    ("frames", "expected"),
    [(5, [0, 1, 2, 3, 4]), (6, [0, 1, 2, 3, 5]), (101, [0, 25, 50, 75, 100])],
)
def test_anchor_indices_use_floor_rule(frames: int, expected: list[int]) -> None:
    assert anchor_indices(frames) == expected


def test_targets_use_task_population_denominator(
    dataset_manifest: dict[str, object], frame_counts: dict[str, int]
) -> None:
    protocol = build_protocol(dataset_manifest, frame_counts)

    first = protocol[0]
    pair = first["videos"][0]["ordered_pairs"][0]
    assert pair["target"] == pytest.approx(
        (pair["end"] - pair["start"]) / first["max_frames"]
    )
