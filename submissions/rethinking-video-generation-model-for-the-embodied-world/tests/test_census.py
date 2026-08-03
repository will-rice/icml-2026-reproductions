from hashlib import sha256
import json

import pytest

from rbench_repro.census import CATEGORY_RULES, run_census


MANIPULATION_PATH = "prompts/common_manipulation_prompts.json"


def test_census_classifies_exactly_five_tasks_and_four_embodiments(acquired_fixture):
    result = run_census(acquired_fixture).to_dict()
    assert [row["partition"] for row in result["categories"]].count("task") == 5
    assert [row["partition"] for row in result["categories"]].count("embodiment") == 4
    assert result["total_records"] == sum(row["record_count"] for row in result["categories"])
    assert result["required_fields"] == ["image_path", "name", "prompt"]
    assert all(row["unexpected_fields"] == ["manipulated_object"] for row in result["categories"])
    assert result["category_mismatches"] == []


def test_category_rules_match_the_nine_pinned_artifact_names():
    assert [(rule.name, rule.partition) for rule in CATEGORY_RULES] == [
        ("common_manipulation", "task"),
        ("long-horizon_planning", "task"),
        ("multi-entity_collaboration", "task"),
        ("spatial_relationship", "task"),
        ("visual_reasoning", "task"),
        ("dual_arm", "embodiment"),
        ("humanoid", "embodiment"),
        ("quad", "embodiment"),
        ("single_arm", "embodiment"),
    ]


def test_census_rejects_missing_category(acquired_fixture):
    acquired_fixture["dataset"].remove("prompts/visual_reasoning_prompts.json")
    with pytest.raises(ValueError, match="missing prompt manifest: visual_reasoning"):
        run_census(acquired_fixture)


def test_census_reports_duplicate_and_normalized_ids_and_broken_references(acquired_fixture):
    acquired_fixture["dataset"].replace_json(
        MANIPULATION_PATH,
        [
            {"name": "shared-id", "prompt": "p", "image_path": "missing.jpg"},
            {
                "name": "shared-id",
                "prompt": "q",
                "image_path": "common_manipulation/0001.jpg",
            },
            {
                "name": " Shared_ID ",
                "prompt": "r",
                "image_path": "common_manipulation/0001.jpg",
            },
        ],
    )
    result = run_census(acquired_fixture).to_dict()
    assert result["duplicate_ids"] == ["shared-id"]
    assert result["normalized_id_collisions"] == [[" Shared_ID ", "shared-id"]]
    assert result["missing_references"] == ["imgs/missing.jpg"]


def test_census_uses_full_tree_metadata_without_fetched_image(acquired_fixture):
    dataset = acquired_fixture["dataset"]
    image = "imgs/common_manipulation/0001.jpg"
    assert image not in {record.path for record in dataset.manifest.files}
    assert image in {entry.path for entry in dataset.manifest.tree}
    assert not (dataset.root / image).exists()
    assert run_census(acquired_fixture).to_dict()["missing_references"] == []


@pytest.mark.parametrize("image_path", ["../escape.jpg", "/absolute.jpg", "a/../../escape.jpg"])
def test_census_rejects_unsafe_relative_image_paths(acquired_fixture, image_path):
    acquired_fixture["dataset"].replace_json(
        MANIPULATION_PATH,
        [{"name": "unsafe", "prompt": "p", "image_path": image_path}],
    )
    result = run_census(acquired_fixture).to_dict()
    assert result["malformed_manifests"][0]["diagnostic"] == (
        "record 0 field image_path is unsafe"
    )
    assert result["reference_checks"] == [
        {
            "exists": True,
            "path": "imgs/dual_arm/0001.jpg",
        },
        {
            "exists": True,
            "path": "imgs/humanoid/0001.jpg",
        },
        {
            "exists": True,
            "path": "imgs/long-horizon_planning/0001.jpg",
        },
        {
            "exists": True,
            "path": "imgs/multi-entity_collaboration/0001.jpg",
        },
        {
            "exists": True,
            "path": "imgs/quad/0001.jpg",
        },
        {
            "exists": True,
            "path": "imgs/single_arm/0001.jpg",
        },
        {
            "exists": True,
            "path": "imgs/spatial_relationship/0001.jpg",
        },
        {
            "exists": True,
            "path": "imgs/visual_reasoning/0001.jpg",
        },
    ]


def test_census_reports_missing_and_unexpected_semantic_fields(acquired_fixture):
    acquired_fixture["dataset"].replace_json(
        MANIPULATION_PATH,
        [
            {
                "name": "m1",
                "image_path": "common_manipulation/0001.jpg",
                "surprise": 1,
            }
        ],
    )
    row = run_census(acquired_fixture).to_dict()["categories"][0]
    assert row["missing_required_fields"] == ["prompt"]
    assert row["unexpected_fields"] == ["surprise"]


@pytest.mark.parametrize(
    ("payload", "diagnostic"),
    [
        (b"{", "invalid JSON"),
        (b"{}", "top level is not a list"),
        (b"[1]", "record 0 is not an object"),
    ],
)
def test_census_preserves_malformed_hash_with_sanitized_diagnostic(
    acquired_fixture, payload, diagnostic
):
    acquired_fixture["dataset"].replace_bytes(MANIPULATION_PATH, payload)
    result = run_census(acquired_fixture).to_dict()
    assert result["malformed_manifests"] == [
        {
            "diagnostic": diagnostic,
            "path": MANIPULATION_PATH,
            "sha256": sha256(payload).hexdigest(),
        }
    ]
    assert result["categories"][0]["record_count"] == 0
    assert result["categories"][0]["missing_required_fields"] == [
        "image_path",
        "name",
        "prompt",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [("name", 1), ("prompt", ""), ("image_path", "   ")],
)
def test_census_reports_invalid_required_semantic_types(
    acquired_fixture, field, value
):
    record = {
        "name": "m1",
        "prompt": "p",
        "image_path": "common_manipulation/0001.jpg",
    }
    record[field] = value
    acquired_fixture["dataset"].replace_json(MANIPULATION_PATH, [record])
    result = run_census(acquired_fixture).to_dict()
    assert result["malformed_manifests"][0]["diagnostic"] == (
        f"record 0 field {field} is not a nonempty string"
    )


@pytest.mark.parametrize(
    "location", ["eval", "eval_embodiment", "shell", "leaderboard"]
)
def test_census_reports_unknown_extra_categories(acquired_fixture, location):
    if location == "eval":
        acquired_fixture["revidgen"].replace_bytes(
            "eval/5_tasks/summary_scores.py",
            b'TASK_TYPES = ["Common Manipulation", "Unknown Task"]\n',
        )
        expected = "eval_tasks extra: Unknown Task"
    elif location == "eval_embodiment":
        acquired_fixture["revidgen"].replace_bytes(
            "eval/4_embodiments/summary_scores.py",
            b'ROBOT_TYPES = ["Dual Arm", "Unknown Robot"]\n',
        )
        expected = "eval_embodiments extra: Unknown Robot"
    elif location == "shell":
        source = acquired_fixture["revidgen"]
        path = "scripts/rbench_eval_5tasks.sh"
        source.replace_bytes(
            path,
            b'task_cfg=("common_manipulation:100" "unknown_shell_task:3")\n'
            b'for cfg in "${task_cfg[@]}"; do\n'
            b'  IFS=: read -r TASK_TYPE COUNT <<< "$cfg"\n'
            b'  python "eval/5_tasks/${TASK_TYPE}.py" --count "$COUNT"\n'
            b'done\n',
        )
        expected = "shell_tasks extra: unknown_shell_task"
    else:
        source = acquired_fixture["paper"]
        row = json.loads((source.root / "leaderboard.json").read_bytes())[0]
        row["unknown_column"] = 1.0
        source.replace_json("leaderboard.json", [row])
        expected = "leaderboard_paper extra: unknown_column"
    assert expected in run_census(acquired_fixture).to_dict()["category_mismatches"]


def test_census_rejects_task_cfg_disconnected_from_evaluator_loop(acquired_fixture):
    acquired_fixture["revidgen"].replace_bytes(
        "scripts/rbench_eval_5tasks.sh",
        b'task_cfg=("common_manipulation:100")\n'
        b'python "eval/5_tasks/${TASK_TYPE}.py"\n',
    )
    category_sets = run_census(acquired_fixture).to_dict()["category_sets"]
    assert category_sets["shell_tasks"] == []


def test_census_rejects_task_cfg_when_task_type_is_rebound(acquired_fixture):
    acquired_fixture["revidgen"].replace_bytes(
        "scripts/rbench_eval_5tasks.sh",
        b'task_cfg=("common_manipulation:100")\n'
        b'for cfg in "${task_cfg[@]}"; do\n'
        b'  IFS=: read -r TASK_TYPE COUNT <<< "$cfg"\n'
        b'  TASK_TYPE=visual_reasoning\n'
        b'  python "eval/5_tasks/${TASK_TYPE}.py"\n'
        b'done\n',
    )
    category_sets = run_census(acquired_fixture).to_dict()["category_sets"]
    assert category_sets["shell_tasks"] == []


def test_census_uses_only_task_cfg_assignment_overriding_stale_values(acquired_fixture):
    acquired_fixture["revidgen"].replace_bytes(
        "scripts/rbench_eval_5tasks.sh",
        b'task_cfg=("stale_task:1")\n'
        b'task_cfg=("common_manipulation:100")\n'
        b'for cfg in "${task_cfg[@]}"; do\n'
        b'  IFS=: read -r TASK_TYPE COUNT <<< "$cfg"\n'
        b'  python "eval/5_tasks/${TASK_TYPE}.py"\n'
        b'done\n',
    )
    category_sets = run_census(acquired_fixture).to_dict()["category_sets"]
    assert category_sets["shell_tasks"] == ["common_manipulation"]


def test_census_ignores_task_cfg_assignment_after_connected_loop(acquired_fixture):
    acquired_fixture["revidgen"].replace_bytes(
        "scripts/rbench_eval_5tasks.sh",
        b'task_cfg=("common_manipulation:100")\n'
        b'for cfg in "${task_cfg[@]}"; do\n'
        b'  IFS=: read -r TASK_TYPE COUNT <<< "$cfg"\n'
        b'  python "eval/5_tasks/${TASK_TYPE}.py"\n'
        b'done\n'
        b'task_cfg=("post_loop_task:1")\n',
    )
    category_sets = run_census(acquired_fixture).to_dict()["category_sets"]
    assert category_sets["shell_tasks"] == ["common_manipulation"]


def test_census_ignores_disconnected_task_types_array(acquired_fixture):
    acquired_fixture["revidgen"].replace_bytes(
        "scripts/rbench_eval_5tasks.sh",
        b'task_types=("Common Manipulation" "Visual Reasoning")\n',
    )
    category_sets = run_census(acquired_fixture).to_dict()["category_sets"]
    assert category_sets["shell_tasks"] == []


def test_census_ignores_stray_literal_task_evaluator_path(acquired_fixture):
    acquired_fixture["revidgen"].replace_bytes(
        "scripts/rbench_eval_5tasks.sh",
        b"python eval/5_tasks/common_manipulation.py\n",
    )
    category_sets = run_census(acquired_fixture).to_dict()["category_sets"]
    assert category_sets["shell_tasks"] == []


def test_census_ignores_robot_types_not_bound_to_loop(acquired_fixture):
    acquired_fixture["revidgen"].replace_bytes(
        "scripts/rbench_eval_4embodiments.sh",
        b'robot_types=("Dual Arm" "Humanoid Robot")\n'
        b'python eval/4_embodiments/summary_scores.py --robot_type "$robot_type"\n',
    )
    category_sets = run_census(acquired_fixture).to_dict()["category_sets"]
    assert category_sets["shell_embodiments"] == []


@pytest.mark.parametrize(
    ("source", "path", "payload", "expected"),
    [
        (
            "revidgen",
            "scripts/rbench_eval_5tasks.sh",
            b'task_cfg=("common_manipulation:100")\n'
            b'for cfg in "${task_cfg[@]}"; do\n'
            b'  IFS=: read -r TASK_TYPE COUNT <<< "$cfg"\n'
            b'  python "eval/5_tasks/${TASK_TYPE}.py"\n'
            b'done\n',
            "shell_tasks missing",
        ),
        (
            "paper",
            "leaderboard.json",
            b'[{"model":"fixture","Common Manipulation":1}]',
            "leaderboard_paper missing",
        ),
    ],
)
def test_census_reports_sorted_cross_source_category_mismatches(
    acquired_fixture, source, path, payload, expected
):
    acquired_fixture[source].replace_bytes(path, payload)
    result = run_census(acquired_fixture).to_dict()
    assert result["category_mismatches"] == sorted(result["category_mismatches"])
    assert any(item.startswith(expected) for item in result["category_mismatches"])


def test_census_exposes_independent_cross_source_category_sets(acquired_fixture):
    category_sets = run_census(acquired_fixture).to_dict()["category_sets"]
    assert sorted(category_sets) == [
        "eval_embodiments",
        "eval_tasks",
        "leaderboard_current",
        "leaderboard_paper",
        "shell_embodiments",
        "shell_tasks",
    ]
    assert all(values == sorted(values) for values in category_sets.values())


def test_pinned_source_shaped_category_sets_normalize_to_same_categories(acquired_fixture):
    category_sets = run_census(acquired_fixture).to_dict()["category_sets"]
    tasks = [
        "common_manipulation",
        "long-horizon_planning",
        "multi-entity_collaboration",
        "spatial_relationship",
        "visual_reasoning",
    ]
    embodiments = ["dual_arm", "humanoid", "quad", "single_arm"]
    assert category_sets["eval_tasks"] == tasks
    assert category_sets["shell_tasks"] == tasks
    assert category_sets["eval_embodiments"] == embodiments
    assert category_sets["shell_embodiments"] == embodiments
    assert category_sets["leaderboard_paper"] == sorted(tasks + embodiments)
    assert category_sets["leaderboard_current"] == sorted(tasks + embodiments)
