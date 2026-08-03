import ast
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import PurePosixPath
import re
import shlex
from typing import Literal

from rbench_repro.acquisition import AcquiredSource


@dataclass(frozen=True, slots=True)
class CategoryRule:
    name: str
    partition: Literal["task", "embodiment"]
    prompt_path: str
    eval_path: str
    leaderboard_column: str


CATEGORY_RULES = (
    CategoryRule(
        "common_manipulation",
        "task",
        "prompts/common_manipulation_prompts.json",
        "eval/5_tasks/common_manipulation.py",
        "Common Manipulation",
    ),
    CategoryRule(
        "long-horizon_planning",
        "task",
        "prompts/long-horizon_planning_prompts.json",
        "eval/5_tasks/long-horizon_planning.py",
        "Long-Horizon Planning",
    ),
    CategoryRule(
        "multi-entity_collaboration",
        "task",
        "prompts/multi-entity_collaboration_prompts.json",
        "eval/5_tasks/multi-entity_collaboration.py",
        "Multi-Entity Collaboration",
    ),
    CategoryRule(
        "spatial_relationship",
        "task",
        "prompts/spatial_relationship_prompts.json",
        "eval/5_tasks/spatial_relationship.py",
        "Spatial Relationship",
    ),
    CategoryRule(
        "visual_reasoning",
        "task",
        "prompts/visual_reasoning_prompts.json",
        "eval/5_tasks/visual_reasoning.py",
        "Visual Reasoning",
    ),
    CategoryRule(
        "dual_arm",
        "embodiment",
        "prompts/dual_arm_prompts.json",
        "eval/4_embodiments/summary_scores.py",
        "Dual Arm",
    ),
    CategoryRule(
        "humanoid",
        "embodiment",
        "prompts/humanoid_prompts.json",
        "eval/4_embodiments/summary_scores.py",
        "Humanoid Robot",
    ),
    CategoryRule(
        "quad",
        "embodiment",
        "prompts/quad_prompts.json",
        "eval/4_embodiments/summary_scores.py",
        "Quadruped Robot",
    ),
    CategoryRule(
        "single_arm",
        "embodiment",
        "prompts/single_arm_prompts.json",
        "eval/4_embodiments/summary_scores.py",
        "Single Arm",
    ),
)

IDENTIFIER_FIELD = "name"
IMAGE_REFERENCE_FIELD = "image_path"
EXPLICIT_REQUIRED_FIELDS = frozenset({IDENTIFIER_FIELD, "prompt", IMAGE_REFERENCE_FIELD})
LEADERBOARD_NON_CATEGORY_FIELDS = frozenset({"avg", "model"})
CATEGORY_ALIASES = {
    "Common Manipulation": "common_manipulation",
    "Long-Horizon Planning": "long-horizon_planning",
    "Long Horizon Planning": "long-horizon_planning",
    "Multi-Entity Collaboration": "multi-entity_collaboration",
    "Multi Entity Collaboration": "multi-entity_collaboration",
    "Spatial Relationship": "spatial_relationship",
    "Visual Reasoning": "visual_reasoning",
    "Dual Arm": "dual_arm",
    "Dual Arm Robot": "dual_arm",
    "Humanoid": "humanoid",
    "Humanoid Robot": "humanoid",
    "Quad": "quad",
    "Quad Robot": "quad",
    "Quadruped": "quad",
    "Quadruped Robot": "quad",
    "Single Arm": "single_arm",
    "Single Arm Robot": "single_arm",
}
EVAL_AUXILIARY_STEMS = frozenset(
    {
        "1_robot_subject_stability",
        "2_physical_plausibility",
        "3_task_adherence_consistency",
        "4_create_meta_info",
        "5_motion_amplitude",
        "6_motion_smoothness",
        "7_motion_total_score",
        "8_summarize_robot_results",
        "summarize_i2v_results",
        "summary_scores",
    }
)


@dataclass(frozen=True, slots=True)
class CensusResult:
    categories: tuple[dict[str, object], ...]
    total_records: int
    required_fields: tuple[str, ...]
    duplicate_ids: tuple[str, ...]
    normalized_id_collisions: tuple[tuple[str, ...], ...]
    missing_references: tuple[str, ...]
    reference_checks: tuple[dict[str, object], ...]
    malformed_manifests: tuple[dict[str, str], ...]
    category_sets: dict[str, tuple[str, ...]]
    category_mismatches: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "categories": [dict(row) for row in self.categories],
            "category_mismatches": list(self.category_mismatches),
            "category_sets": {
                name: list(values) for name, values in sorted(self.category_sets.items())
            },
            "duplicate_ids": list(self.duplicate_ids),
            "malformed_manifests": [dict(item) for item in self.malformed_manifests],
            "missing_references": list(self.missing_references),
            "normalized_id_collisions": [
                list(group) for group in self.normalized_id_collisions
            ],
            "reference_checks": [dict(item) for item in self.reference_checks],
            "required_fields": list(self.required_fields),
            "total_records": self.total_records,
        }


def run_census(sources: dict[str, AcquiredSource]) -> CensusResult:
    dataset = source_by_label(sources, "rbench-dataset")
    payload_paths = manifested_paths(dataset)
    dataset_paths = tree_file_paths(dataset)
    parsed = []
    malformed = []

    for rule in CATEGORY_RULES:
        if rule.prompt_path not in payload_paths:
            raise ValueError(f"missing prompt manifest: {rule.name}")
        payload = read_manifested_bytes(dataset, rule.prompt_path)
        records, diagnostics = parse_manifest(payload)
        parsed.append((rule, records))
        malformed.extend(
            {
                "diagnostic": diagnostic,
                "path": rule.prompt_path,
                "sha256": sha256(payload).hexdigest(),
            }
            for diagnostic in diagnostics
        )

    required_fields = tuple(sorted(EXPLICIT_REQUIRED_FIELDS))
    identifiers = []
    references = []
    rows = []
    for rule, records in parsed:
        identifiers.extend(
            record[IDENTIFIER_FIELD]
            for record in records
            if is_nonempty_string(record.get(IDENTIFIER_FIELD))
        )
        references.extend(
            resolved
            for record in records
            if (resolved := resolve_image_path(record.get(IMAGE_REFERENCE_FIELD))) is not None
        )
        present_fields = set().union(*(set(record) for record in records)) if records else set()
        rows.append(
            {
                "eval_path": rule.eval_path,
                "leaderboard_column": rule.leaderboard_column,
                "missing_required_fields": sorted(
                    field
                    for field in required_fields
                    if not records
                    or any(
                        field not in record
                        or (
                            field in EXPLICIT_REQUIRED_FIELDS
                            and not is_nonempty_string(record[field])
                        )
                        for record in records
                    )
                ),
                "name": rule.name,
                "partition": rule.partition,
                "prompt_path": rule.prompt_path,
                "record_count": len(records),
                "unexpected_fields": sorted(present_fields - set(required_fields)),
            }
        )

    id_counts = Counter(identifiers)
    normalized_groups: dict[str, set[str]] = {}
    for identifier in identifiers:
        normalized_groups.setdefault(normalized_name(identifier), set()).add(identifier)
    normalized_collisions = tuple(
        tuple(sorted(group))
        for group in sorted(normalized_groups.values(), key=lambda values: sorted(values))
        if len(group) > 1
    )
    reference_checks = tuple(
        {"exists": reference in dataset_paths, "path": reference}
        for reference in sorted(set(references))
    )
    category_sets = cross_source_category_sets(sources)
    mismatches = category_mismatches(category_sets)
    return CensusResult(
        categories=tuple(rows),
        total_records=sum(len(records) for _rule, records in parsed),
        required_fields=required_fields,
        duplicate_ids=tuple(
            sorted(identifier for identifier, count in id_counts.items() if count > 1)
        ),
        normalized_id_collisions=normalized_collisions,
        missing_references=tuple(
            item["path"] for item in reference_checks if not item["exists"]
        ),
        reference_checks=reference_checks,
        malformed_manifests=tuple(
            sorted(malformed, key=lambda item: (item["path"], item["diagnostic"]))
        ),
        category_sets=category_sets,
        category_mismatches=mismatches,
    )


def source_by_label(sources: dict[str, AcquiredSource], label: str) -> AcquiredSource:
    matches = [source for source in sources.values() if source.manifest.label == label]
    if len(matches) != 1:
        raise ValueError(f"required source unavailable: {label}")
    return matches[0]


def manifested_paths(source: AcquiredSource) -> frozenset[str]:
    return frozenset(record.path for record in source.manifest.files)


def tree_file_paths(source: AcquiredSource) -> frozenset[str]:
    return frozenset(entry.path for entry in source.manifest.tree if entry.kind == "file")


def read_manifested_bytes(source: AcquiredSource, path: str) -> bytes:
    if path not in manifested_paths(source):
        raise ValueError(f"required artifact unavailable: {path}")
    try:
        return (source.root / PurePosixPath(path)).read_bytes()
    except OSError:
        raise ValueError(f"required artifact unavailable: {path}") from None


def parse_manifest(payload: bytes) -> tuple[list[dict[str, object]], list[str]]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return [], ["invalid JSON"]
    if not isinstance(value, list):
        return [], ["top level is not a list"]
    records = []
    diagnostics = []
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            diagnostics.append(f"record {index} is not an object")
        else:
            records.append(record)
            for field in sorted(EXPLICIT_REQUIRED_FIELDS):
                if field not in record:
                    diagnostics.append(f"record {index} field {field} is missing")
                elif not is_nonempty_string(record[field]):
                    diagnostics.append(
                        f"record {index} field {field} is not a nonempty string"
                    )
            image_path = record.get(IMAGE_REFERENCE_FIELD)
            if is_nonempty_string(image_path) and resolve_image_path(image_path) is None:
                diagnostics.append(f"record {index} field image_path is unsafe")
    return records, diagnostics


def is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def resolve_image_path(value: object) -> str | None:
    if not is_nonempty_string(value) or not isinstance(value, str) or "\\" in value:
        return None
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != value:
        return None
    return (PurePosixPath("imgs") / relative).as_posix()


def normalized_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def cross_source_category_sets(
    sources: dict[str, AcquiredSource],
) -> dict[str, tuple[str, ...]]:
    revidgen = source_by_label(sources, "revidgen")
    paper = source_by_label(sources, "rbench-leaderboard-paper-era")
    current = source_by_label(sources, "rbench-leaderboard-current")
    shell_task_text = read_manifested_bytes(revidgen, "scripts/rbench_eval_5tasks.sh").decode(
        errors="replace"
    )
    shell_embodiment_text = read_manifested_bytes(
        revidgen, "scripts/rbench_eval_4embodiments.sh"
    ).decode(errors="replace")
    return {
        "eval_embodiments": eval_categories(revidgen, "embodiment"),
        "eval_tasks": eval_categories(revidgen, "task"),
        "leaderboard_current": leaderboard_categories(current),
        "leaderboard_paper": leaderboard_categories(paper),
        "shell_embodiments": shell_categories(shell_embodiment_text, "embodiment"),
        "shell_tasks": shell_categories(shell_task_text, "task"),
    }


def eval_categories(
    source: AcquiredSource, partition: Literal["task", "embodiment"]
) -> tuple[str, ...]:
    directory = "5_tasks" if partition == "task" else "4_embodiments"
    prefix = f"eval/{directory}/"
    categories = {
        category_name(PurePosixPath(path).stem)
        for path in tree_file_paths(source)
        if path.startswith(prefix)
        and path.endswith(".py")
        and PurePosixPath(path).stem not in EVAL_AUXILIARY_STEMS
    }
    for path in sorted(path for path in manifested_paths(source) if path.startswith(prefix)):
        try:
            tree = ast.parse(read_manifested_bytes(source, path))
        except (SyntaxError, ValueError):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            partition_tokens = (
                ("task",) if partition == "task" else ("robot", "embodiment")
            )
            if not any(
                "categor" in name.casefold()
                or any(token in name.casefold() for token in partition_tokens)
                for name in names
            ):
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, (list, tuple, set)):
                categories.update(category_name(item) for item in value if isinstance(item, str))
    return tuple(sorted(categories))


def shell_categories(
    text: str, partition: Literal["task", "embodiment"]
) -> tuple[str, ...]:
    assignment_pattern = re.compile(
        r"(?ms)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\((.*?)\)"
    )
    if partition == "task":
        return tuple(sorted(connected_task_cfg_categories(text, assignment_pattern)))
    return tuple(sorted(connected_robot_type_categories(text, assignment_pattern)))


def connected_task_cfg_categories(
    text: str, assignment_pattern: re.Pattern[str]
) -> set[str]:
    assignments = [
        match
        for match in assignment_pattern.finditer(text)
        if match.group(1).casefold() == "task_cfg"
    ]
    loop_pattern = re.compile(
        r'''(?ms)for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+
        ["']?\$\{task_cfg\[@\]\}["']?\s*;\s*do\s*(.*?)^\s*done\b''',
        re.VERBOSE,
    )
    categories = set()
    for loop in loop_pattern.finditer(text):
        loop_variable, body = loop.groups()
        if not task_cfg_loop_is_connected(loop_variable, body):
            continue
        preceding = [assignment for assignment in assignments if assignment.end() <= loop.start()]
        if not preceding:
            continue
        assignment = preceding[-1]
        try:
            values = shlex.split(assignment.group(2), comments=True, posix=True)
        except ValueError:
            continue
        for value in values:
            task_name, separator, count = value.rpartition(":")
            if separator and task_name and count.isdigit():
                categories.add(category_name(task_name))
    return categories


def task_cfg_loop_is_connected(loop_variable: str, body: str) -> bool:
    invocation_pattern = re.compile(r"eval/5_tasks/\$\{TASK_TYPE\}\.py")
    rebound_pattern = re.compile(r"(?m)^\s*TASK_TYPE\s*=")
    variable = re.escape(loop_variable)
    variable_rebound_pattern = re.compile(rf"(?m)^\s*{variable}\s*=")
    derivation_patterns = (
        re.compile(
            rf'''(?mx)^\s*IFS\s*=\s*["']?:["']?\s+read\s+
            (?:-[A-Za-z]+\s+)*TASK_TYPE(?:\s+[A-Za-z_][A-Za-z0-9_]*)*\s+
            <<<\s*["']?(?:\$\{{{variable}\}}|\${variable})["']?\s*$'''
        ),
        re.compile(
            rf'''(?mx)^\s*TASK_TYPE\s*=\s*["']?
            \$\{{{variable}%%:\*\}}["']?\s*$'''
        ),
    )
    derivations = sorted(
        (match for pattern in derivation_patterns for match in pattern.finditer(body)),
        key=lambda match: match.start(),
    )
    for derivation in derivations:
        if variable_rebound_pattern.search(body, 0, derivation.start()):
            continue
        for invocation in invocation_pattern.finditer(body, derivation.end()):
            if not rebound_pattern.search(body, derivation.end(), invocation.start()):
                return True
    return False


def connected_robot_type_categories(
    text: str, assignment_pattern: re.Pattern[str]
) -> set[str]:
    assignments = [
        match
        for match in assignment_pattern.finditer(text)
        if match.group(1).casefold() == "robot_types"
    ]
    loop_pattern = re.compile(
        r'''(?ms)for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+
        ["']?\$\{robot_types\[@\]\}["']?\s*;\s*do\s*(.*?)^\s*done\b''',
        re.VERBOSE,
    )
    categories = set()
    for loop in loop_pattern.finditer(text):
        loop_variable, body = loop.groups()
        variable = re.escape(loop_variable)
        invocation_pattern = re.compile(
            rf'''(?m)^.*eval/4_embodiments/[A-Za-z0-9_-]+\.py.*
            --robot_type\s+["']?(?:\$\{{{variable}\}}|\${variable})["']?.*$''',
            re.VERBOSE,
        )
        invocation = invocation_pattern.search(body)
        rebound_pattern = re.compile(rf"(?m)^\s*{variable}\s*=")
        if invocation is None or rebound_pattern.search(body, 0, invocation.start()):
            continue
        preceding = [assignment for assignment in assignments if assignment.end() <= loop.start()]
        if not preceding:
            continue
        try:
            values = shlex.split(preceding[-1].group(2), comments=True, posix=True)
        except ValueError:
            continue
        categories.update(category_name(value) for value in values)
    return categories


def category_name(value: str) -> str:
    candidate = value.strip()
    expected = {
        normalized_name(alias): canonical
        for alias, canonical in CATEGORY_ALIASES.items()
    }
    expected.update(
        {
            normalized_name(name): rule.name
            for rule in CATEGORY_RULES
            for name in (rule.name, rule.leaderboard_column)
        }
    )
    return expected.get(normalized_name(candidate), candidate)


def leaderboard_categories(source: AcquiredSource) -> tuple[str, ...]:
    payload = read_manifested_bytes(source, "leaderboard.json")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ()
    rows = value if isinstance(value, list) else []
    fields = {
        category_name(key)
        for row in rows
        if isinstance(row, dict)
        for key in row
        if isinstance(key, str) and key not in LEADERBOARD_NON_CATEGORY_FIELDS
    }
    return tuple(sorted(fields))


def category_mismatches(category_sets: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    tasks = {rule.name for rule in CATEGORY_RULES if rule.partition == "task"}
    embodiments = {rule.name for rule in CATEGORY_RULES if rule.partition == "embodiment"}
    all_categories = tasks | embodiments
    expected = {
        "eval_embodiments": embodiments,
        "eval_tasks": tasks,
        "leaderboard_current": all_categories,
        "leaderboard_paper": all_categories,
        "shell_embodiments": embodiments,
        "shell_tasks": tasks,
    }
    mismatches = []
    for name, expected_values in expected.items():
        actual = set(category_sets[name])
        missing = sorted(expected_values - actual)
        extra = sorted(actual - expected_values)
        if missing:
            mismatches.append(f"{name} missing: {','.join(missing)}")
        if extra:
            mismatches.append(f"{name} extra: {','.join(extra)}")
    return tuple(sorted(mismatches))
