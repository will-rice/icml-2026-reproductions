from dataclasses import dataclass, replace
import json
from pathlib import Path

import pytest

from rbench_repro.acquisition import AcquiredSource, FileRecord, SourceManifest, TreeEntry
from rbench_repro.model import sha256_bytes


@dataclass
class MemoryReader:
    resolved: str
    entries: list[tuple[str, str, int]]
    payloads: dict[str, bytes]
    read_calls: int = 0
    tree_calls: int = 0

    def resolve(self, spec):
        return self.resolved

    def tree(self, spec, revision):
        from rbench_repro.acquisition import TreeEntry

        self.tree_calls += 1
        return tuple(TreeEntry(*entry) for entry in self.entries)

    def read(self, spec, revision, path, timeout_seconds):
        assert timeout_seconds == 30
        self.read_calls += 1
        return self.payloads[path]


@pytest.fixture
def source_spec():
    from rbench_repro.acquisition import SourceSpec

    return SourceSpec(
        label="fixture",
        kind="dataset",
        repo_id="owner/repo",
        canonical_url="https://example.test/owner/repo",
        requested_revision="a" * 40,
        allowlist=("README.md", "prompts/example.json"),
        license_id="cc-by-4.0",
        license_source="README.md",
        redistributable=True,
        command="hf download owner/repo --revision " + "a" * 40,
    )


@pytest.fixture
def source_reader(source_spec):
    payloads = {
        "README.md": b"fixture readme\n",
        "prompts/example.json": b'{"text":"source text sentinel"}\n',
    }
    return MemoryReader(
        resolved=source_spec.requested_revision,
        entries=[(path, "file", len(payload)) for path, payload in payloads.items()],
        payloads=payloads,
    )


@dataclass
class MutableAcquiredSource:
    acquired: AcquiredSource

    @property
    def root(self):
        return self.acquired.root

    @property
    def manifest(self):
        return self.acquired.manifest

    def remove(self, path):
        (self.root / path).unlink()
        self.acquired = replace(
            self.acquired,
            manifest=replace(
                self.manifest,
                files=tuple(record for record in self.manifest.files if record.path != path),
                tree=tuple(entry for entry in self.manifest.tree if entry.path != path),
            ),
        )

    def replace_bytes(self, path, payload):
        (self.root / path).write_bytes(payload)
        replacement = FileRecord(path=path, bytes=len(payload), sha256=sha256_bytes(payload))
        files = tuple(replacement if record.path == path else record for record in self.manifest.files)
        tree = tuple(
            TreeEntry(path=path, kind="file", size=len(payload)) if entry.path == path else entry
            for entry in self.manifest.tree
        )
        self.acquired = replace(
            self.acquired, manifest=replace(self.manifest, files=files, tree=tree)
        )

    def replace_json(self, path, value):
        self.replace_bytes(path, json.dumps(value).encode())

def acquired_source(
    root: Path,
    label: str,
    payloads: dict[str, bytes],
    metadata_only: tuple[TreeEntry, ...] = (),
) -> MutableAcquiredSource:
    source_root = root / label
    for path, payload in payloads.items():
        destination = source_root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    manifest = SourceManifest(
        label=label,
        kind=(
            "dataset"
            if label == "rbench-dataset"
            else "git"
            if label == "revidgen"
            else "space"
        ),
        repo_id=f"fixture/{label}",
        canonical_url=f"https://example.test/{label}",
        requested_revision="a" * 40,
        resolved_revision="a" * 40,
        acquired_at="2026-07-25T00:00:00+00:00",
        license_id="fixture",
        license_source="fixture",
        redistributable=label == "rbench-dataset",
        command="fixture",
        files=tuple(
            FileRecord(path=path, bytes=len(payload), sha256=sha256_bytes(payload))
            for path, payload in payloads.items()
        ),
        tree=tuple(
            sorted(
                (
                    *(TreeEntry(path=path, kind="file", size=len(payload)) for path, payload in payloads.items()),
                    *metadata_only,
                ),
                key=lambda entry: entry.path,
            )
        ),
    )
    return MutableAcquiredSource(AcquiredSource(manifest=manifest, root=source_root))


@pytest.fixture
def acquired_fixture(tmp_path):
    tasks = (
        ("common_manipulation", "Common Manipulation"),
        ("long-horizon_planning", "Long-Horizon Planning"),
        ("multi-entity_collaboration", "Multi-Entity Collaboration"),
        ("spatial_relationship", "Spatial Relationship"),
        ("visual_reasoning", "Visual Reasoning"),
    )
    embodiments = (
        ("dual_arm", "Dual Arm"),
        ("humanoid", "Humanoid Robot"),
        ("quad", "Quadruped Robot"),
        ("single_arm", "Single Arm"),
    )
    categories = tasks + embodiments
    prompts = {
        f"prompts/{name}_prompts.json": json.dumps(
            [
                {
                    "name": f"{name}-0001",
                    "prompt": f"Prompt for {label}",
                    "image_path": f"{name}/0001.jpg",
                    "manipulated_object": "red block",
                }
            ]
        ).encode()
        for name, label in categories
    }
    def _task_evaluator(name: str, label: str) -> bytes:
        identifier = f"{name}_score"
        return (
            f'METRIC_IDENTIFIER = "{identifier}"\n'
            f"DIMENSIONS = {list(LEADERBOARD_DIMENSIONS)!r}\n"
            "\n"
            f"def parse_{name}(payload):\n"
            f"    value = payload[\"{identifier}\"]\n"
            f"    return min(1.0, max(0.0, float(value)))\n"
            "\n"
            f"def aggregate_{name}(values):\n"
            "    return sum(values) / len(values)\n"
            "\n"
            f"def evaluate(payloads):\n"
            f"    scores = [parse_{name}(payload) for payload in payloads]\n"
            f'    return {{METRIC_IDENTIFIER: aggregate_{name}(scores), "dimensions": DIMENSIONS}}\n'
        ).encode()
    eval_payloads = {
        **{f"eval/5_tasks/{name}.py": _task_evaluator(name, _label) for name, _label in tasks},
        "eval/5_tasks/summary_scores.py": (
            "TASK_TYPES = " + repr([label for _name, label in tasks]) + "\n"
        ).encode(),
        "eval/4_embodiments/summary_scores.py": (
            "ROBOT_TYPES = " + repr([label for _name, label in embodiments]) + "\n"
        ).encode(),
        "scripts/rbench_eval_5tasks.sh": (
            "task_cfg=(\n"
            + "".join(f'  "{name}:100"\n' for name, _label in tasks)
            + ")\n"
            + 'for cfg in "${task_cfg[@]}"; do\n'
            + "  IFS=: read -r TASK_TYPE COUNT <<< \"$cfg\"\n"
            + '  python "eval/5_tasks/${TASK_TYPE}.py" --count "$COUNT"\n'
            + "done\n"
        ).encode(),
        "scripts/rbench_eval_4embodiments.sh": (
            "robot_types=("
            + " ".join(f'"{label}"' for _name, label in embodiments)
            + ")\n"
            + 'for robot_type in "${robot_types[@]}"; do\n'
            + '  python eval/4_embodiments/summary_scores.py --robot_type "$robot_type"\n'
            + "done\n"
        ).encode(),
    }
    columns = {label: 1.0 for _name, label in categories}
    leaderboard = json.dumps([{"model": "fixture", **columns}]).encode()
    return {
        "dataset": acquired_source(
            tmp_path,
            "rbench-dataset",
            prompts,
            metadata_only=tuple(
                TreeEntry(path=f"imgs/{name}/0001.jpg", kind="file", size=22_000_000_000)
                for name, _label in categories
            ),
        ),
        "revidgen": acquired_source(tmp_path, "revidgen", eval_payloads),
        "paper": acquired_source(
            tmp_path,
            "rbench-leaderboard-paper-era",
            {"leaderboard.json": leaderboard},
        ),
        "current": acquired_source(
            tmp_path,
            "rbench-leaderboard-current",
            {"leaderboard.json": leaderboard},
        ),
    }


LEADERBOARD_DIMENSIONS = (
    "Common Manipulation",
    "Long-Horizon Planning",
    "Multi-Entity Collaboration",
    "Spatial Relationship",
    "Visual Reasoning",
    "Dual Arm",
    "Humanoid Robot",
    "Quadruped Robot",
    "Single Arm",
)
PAPER_MODELS = (
    "Wan2.2_A14B",
    "HunyuanVideo 1.5",
    "LongCat-Video",
    "Wan2.1_14B",
    "LTX-2",
    "Wan2.2_5B",
    "SkyReels",
    "LTX-Video",
    "FramePack",
    "HunyuanVideo",
    "CogVideoX_5B",
    "Wan 2.6",
    "Seedance 1.5 pro",
    "Wan 2.5",
    "Hailuo v2",
    "Veo 3",
    "Seedance 1.0",
    "Kling 2.6 pro",
    "Sora v2 Pro#",
    "Sora v1",
    "Cosmos 2.5",
    "DreamGen(gr1)",
    "DreamGen(droid)",
    "Vidar",
    "UnifoLM-WMA-0",
)
PREPENDED_MODELS = ("LingBot-Video", "Cosmos3-Nano", "Cosmos3-Super")


def leaderboard_row(name, value="1.000"):
    return {"model": name, **dict.fromkeys(LEADERBOARD_DIMENSIONS, value), "avg": value}


@pytest.fixture
def formula_sources():
    columns = repr(list(LEADERBOARD_DIMENSIONS))
    utils_source = (
        f"DIMENSION_COLUMNS = {columns}\n"
        "def leaderboard_average(row):\n"
        "    values = [round(row[column], 4) for column in DIMENSION_COLUMNS]\n"
        "    return round(sum(values) / len(values), 3)\n"
    ).encode()
    app_source = b"from utils import leaderboard_average\n"
    return utils_source, app_source


@pytest.fixture
def recovered_formula(formula_sources):
    from rbench_repro.leaderboard import derive_formula

    return derive_formula(*formula_sources)


@pytest.fixture
def groups():
    return {"paper_cohort": PAPER_MODELS, "later_additions": PREPENDED_MODELS}


@pytest.fixture
def leaderboard_sources(tmp_path):
    paper_rows = [leaderboard_row(name) for name in PAPER_MODELS]
    later_rows = [leaderboard_row(name) for name in (*PREPENDED_MODELS, *PAPER_MODELS)]
    return {
        "paper": acquired_source(
            tmp_path,
            "rbench-leaderboard-paper-era",
            {"leaderboard.json": json.dumps(paper_rows).encode()},
        ),
        "later": acquired_source(
            tmp_path,
            "rbench-leaderboard-current",
            {
                "leaderboard.json": json.dumps(later_rows).encode(),
                "leaderboard_qwen.json": json.dumps([leaderboard_row("Qwen evaluator")]).encode(),
            },
        ),
    }


@pytest.fixture
def one_row_source(tmp_path):
    return acquired_source(
        tmp_path,
        "rbench-leaderboard-paper-era",
        {"leaderboard.json": json.dumps([leaderboard_row("Model-1")]).encode()},
    )


@pytest.fixture
def source_audit_fixture(tmp_path):
    task_source = b'''METRIC_IDENTIFIER = "task_adherence_score"
DIMENSIONS = ["Common Manipulation", "Long-Horizon Planning", "Multi-Entity Collaboration", "Spatial Relationship", "Visual Reasoning"]
FAILURE_MODES = {"structural_distortion": "structural distortion"}

def parse_structural_distortion(payload):
    value = payload["structural_distortion"]
    return min(1.0, max(0.0, float(value)))

def aggregate_structural_distortion(values):
    return sum(values) / len(values)

def evaluate(payloads):
    scores = [parse_structural_distortion(payload) for payload in payloads]
    return {METRIC_IDENTIFIER: aggregate_structural_distortion(scores), "dimensions": DIMENSIONS}
'''
    embodiment_source = b'''METRIC_IDENTIFIER = "robot_subject_stability"
DIMENSIONS = ["Dual Arm", "Humanoid Robot", "Quadruped Robot", "Single Arm"]

def parse_stability(payload):
    return float(payload["stability"])

def aggregate_stability(values):
    return sum(values) / len(values)

def evaluate(payloads):
    scores = [parse_stability(payload) for payload in payloads]
    return {METRIC_IDENTIFIER: aggregate_stability(scores), "dimensions": DIMENSIONS}
'''
    return {
        "revidgen": acquired_source(
            tmp_path,
            "revidgen",
            {
                "README.md": b"Pinned evaluator fixture.\n",
                "eval/5_tasks/common_manipulation.py": task_source,
                "eval/4_embodiments/1_robot_subject_stability.py": embodiment_source,
                "scripts/rbench_eval_5tasks.sh": (
                    b'task_cfg=("common_manipulation:100")\n'
                    b'for cfg in "${task_cfg[@]}"; do\n'
                    b'  IFS=: read -r TASK_TYPE COUNT <<< "$cfg"\n'
                    b'  python "eval/5_tasks/${TASK_TYPE}.py" --count "$COUNT"\n'
                    b'done\n'
                ),
                "scripts/rbench_eval_4embodiments.sh": b"python eval/4_embodiments/1_robot_subject_stability.py\n",
            },
        ),
        "paper": acquired_source(
            tmp_path,
            "rbench-leaderboard-paper-era",
            {
                "app.py": b'FAILURE_MODES = {"floating_components": "floating components"}\n',
                "utils.py": b"# no failure-mode aggregation\n",
            },
        ),
        "current": acquired_source(
            tmp_path,
            "rbench-leaderboard-current",
            {"app.py": b"# current UI\n", "utils.py": b"# current helpers\n"},
        ),
    }


@pytest.fixture
def readme_only_failure_fixture(tmp_path):
    return {
        "revidgen": acquired_source(
            tmp_path,
            "revidgen",
            {
                "README.md": b"Taxonomy mentions structural distortion.\n",
                "eval/5_tasks/common_manipulation.py": (
                    b"# structural distortion is prose only\n"
                    b"def parse_other(payload):\n    return float(payload['other'])\n"
                    b"def aggregate_other(values):\n    return sum(values) / len(values)\n"
                ),
                "scripts/rbench_eval_5tasks.sh": b"python eval/5_tasks/common_manipulation.py\n",
            },
        ),
        "paper": acquired_source(
            tmp_path,
            "rbench-leaderboard-paper-era",
            {"app.py": b"# no declarations\n"},
        ),
        "current": acquired_source(
            tmp_path,
            "rbench-leaderboard-current",
            {"app.py": b"# no declarations\n"},
        ),
    }


@pytest.fixture
def complete_audit_inputs(acquired_fixture, recovered_formula, groups):
    """Build a full AuditInputs from the existing test fixtures."""
    from rbench_repro.census import run_census
    from rbench_repro.evidence import AuditInputs
    from rbench_repro.leaderboard import (
        audit_leaderboard,
        compare_cohorts,
    )
    from rbench_repro.source_audit import audit_failure_modes, trace_metrics

    sources = {
        s.acquired.manifest.label: s.acquired
        for s in acquired_fixture.values()
    }
    census = run_census(sources)
    metrics = trace_metrics(sources)
    paper_lb = audit_leaderboard(
        acquired_fixture["paper"].acquired,
        "paper-era",
        "leaderboard.json",
        recovered_formula,
        groups,
    )
    current_lb = audit_leaderboard(
        acquired_fixture["current"].acquired,
        "later",
        "leaderboard.json",
        recovered_formula,
        groups,
    )
    comparison = compare_cohorts(paper_lb, current_lb)
    failure_modes = audit_failure_modes(sources)
    return AuditInputs(
        sources=tuple(s.acquired.manifest for s in acquired_fixture.values()),
        census=census,
        metrics=metrics,
        formula=recovered_formula,
        leaderboards=(paper_lb, current_lb),
        comparison=comparison,
        failure_modes=failure_modes,
        category_evidence=census.category_sets,
        package_lock_sha256=sha256_bytes(b"fixture-lock"),
    )
