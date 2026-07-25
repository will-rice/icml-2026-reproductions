"""Static census of datasets and paradigms in the pinned release."""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

CLAIM_ID = "fourteen-dataset-ten-paradigm-curation"

PAPER_DATASETS = {
    "ADFTD": ("adftd", "AdftdBuilder", "alzheimers_disease_recognition"),
    "BCIC-2a": ("bcic_2a", "BCIC2ABuilder", "motor_imagery"),
    "HMC": ("hmc", "HMCBuilder", "sleep_stage_classification"),
    "Mimul-11": ("mimul_11", "Mimul11Builder", "motor_imagery"),
    "PhysioMI": (
        "motor_mv_img",
        "MotorMoveImagineBuilder",
        "motor_imagery",
    ),
    "SEED": ("seed", "SeedBuilder", "emotion_recognition"),
    "SEED-V": ("seed_v", "SeedVBuilder", "emotion_recognition"),
    "SEED-VII": ("seed_vii", "SeedVIIBuilder", "emotion_recognition"),
    "Siena": ("siena_scalp", "SienaScalpBuilder", "seizure_detection"),
    "TUAB": ("tuab", "TuabBuilder", "abnormal_detection"),
    "TUEV": ("tuev", "TuevBuilder", "event_type_classification"),
    "TUSL": ("tusl", "TuslBuilder", "slowing_event_classification"),
    "Things-EEG-2": (
        "things_eeg_2",
        "ThingsEEG2Builder",
        "visual_target_detection",
    ),
    "Workload": ("workload", "WorkloadBuilder", "mental_stress_assessment"),
}

PAPER_PARADIGMS = sorted({entry[2] for entry in PAPER_DATASETS.values()})


def _parse_python(path: Path) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _dataset_selector(wrapper: Path) -> dict[str, str]:
    tree = _parse_python(wrapper)
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DATASET_SELECTOR"
            for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "DATASET_SELECTOR"
        ):
            value = node.value
        if not isinstance(value, ast.Dict):
            continue
        selector: dict[str, str] = {}
        for key, item in zip(value.keys, value.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if isinstance(item, ast.Name):
                    selector[key.value] = item.id
                elif isinstance(item, ast.Attribute):
                    selector[key.value] = item.attr
        return selector
    raise ValueError(f"DATASET_SELECTOR not found in {wrapper}")


def _builder_classes(dataset_dir: Path) -> set[str]:
    classes: set[str] = set()
    for path in sorted(dataset_dir.rglob("*.py")):
        tree = _parse_python(path)
        classes.update(
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        )
    return classes


def _builder_task_types(dataset_dir: Path) -> dict[str, str]:
    """Associate builders with task types declared in their released module."""

    mapping: dict[str, str] = {}
    for path in sorted(dataset_dir.rglob("*.py")):
        tree = _parse_python(path)
        builders = [
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name.endswith("Builder")
        ]
        task_types: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.AnnAssign):
                continue
            if not isinstance(node.target, ast.Name) or node.target.id != "task_type":
                continue
            if isinstance(node.value, ast.Attribute):
                task_types.append(node.value.attr.lower())
            elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                task_types.append(node.value.value.lower())
        if len(set(task_types)) == 1:
            for builder in builders:
                mapping[builder] = task_types[0]
    return mapping


def _fixture_mapping(config_dir: Path) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = {}
    for path in sorted(config_dir.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        dataset = re.search(r"(?m)^\s*dataset:\s*['\"]?([^'\"#\s]+)", text)
        paradigm = re.search(r"(?m)^\s*paradigm:\s*['\"]?([^'\"#\s]+)", text)
        if dataset and paradigm:
            mapping.setdefault(dataset.group(1), set()).add(paradigm.group(1))
    return {name: sorted(values) for name, values in sorted(mapping.items())}


def run_census_audit(snapshot: Path) -> dict:
    """Compute a JSON-serializable structural census from release source files."""

    wrapper = snapshot / "data" / "processor" / "wrapper.py"
    dataset_dir = snapshot / "data" / "dataset"
    config_dir = snapshot / "assets" / "conf"
    selector = _dataset_selector(wrapper)
    builders = _builder_classes(dataset_dir)
    builder_task_types = _builder_task_types(dataset_dir)
    fixture_mapping = _fixture_mapping(config_dir)

    if fixture_mapping:
        mapping = {
            dataset: paradigms
            for dataset, paradigms in fixture_mapping.items()
            if dataset in selector and selector[dataset] in builders
        }
        mapping_source = "released_configuration"
    else:
        mapping = {
            paper_name: [builder_task_types[builder]]
            for paper_name, (selector_key, builder, _) in PAPER_DATASETS.items()
            if selector.get(selector_key) == builder
            and builder in builders
            and builder in builder_task_types
        }
        mapping_source = "released_dataset_task_type"

    paradigms = sorted({item for values in mapping.values() for item in values})
    config_files = [
        path.relative_to(snapshot).as_posix()
        for path in sorted(config_dir.rglob("*"))
        if path.is_file()
    ]
    return {
        "claim_id": CLAIM_ID,
        "kind": "structural_audit",
        "computed": {
            "dataset_count": len(mapping),
            "paradigm_count": len(paradigms),
            "dataset_paradigms": mapping,
            "mapping_source": mapping_source,
            "selector_entry_count": len(selector),
            "builder_class_count": len(builders),
            "config_files": config_files,
        },
        "paper_context": {
            "label": "paper_reported_not_reproduced",
            "dataset_count": 14,
            "paradigm_count": 10,
            "datasets": sorted(PAPER_DATASETS),
            "paradigms": PAPER_PARADIGMS,
        },
    }
