"""
Data Loader Module

Loads and caches experiment data from the output directory format.
Provides Pandas DataFrames and typed data classes to all dashboard pages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import streamlit as st

    _has_streamlit = True
except ImportError:
    _has_streamlit = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PredictionFolderData:
    """Data from a per-prediction artifact folder."""

    prediction_index: int
    prompt_text: Optional[str] = None
    response_text: Optional[str] = None
    response_meta: Optional[Dict[str, Any]] = None
    predicted_ops: List[str] = field(default_factory=list)
    gt_segment: List[str] = field(default_factory=list)
    history_context: List[str] = field(default_factory=list)
    evaluation: Optional[Dict[str, Any]] = None
    acceptance: Optional[Dict[str, Any]] = None
    future_edits: Optional[Dict[str, Any]] = None


@dataclass
class TrajectoryData:
    """All data for a single trajectory."""

    file_label: str
    trajectory_dir: Path
    summary: Dict[str, Any] = field(default_factory=dict)
    predictions: Optional[pd.DataFrame] = None
    timeline: Optional[pd.DataFrame] = None
    final_trajectory: Optional[pd.DataFrame] = None
    has_prediction_folders: bool = False

    def load_predictions(self) -> pd.DataFrame:
        """Load predictions.jsonl into a flattened DataFrame."""
        if self.predictions is not None:
            return self.predictions
        self.predictions = load_predictions_jsonl(
            self.trajectory_dir / "predictions.jsonl"
        )
        return self.predictions

    def load_timeline(self) -> pd.DataFrame:
        """Load timeline.jsonl into a DataFrame."""
        if self.timeline is not None:
            return self.timeline
        self.timeline = load_timeline_jsonl(self.trajectory_dir / "timeline.jsonl")
        return self.timeline

    def load_final_trajectory(self) -> pd.DataFrame:
        """Load final_trajectory.jsonl into a DataFrame."""
        if self.final_trajectory is not None:
            return self.final_trajectory
        path = self.trajectory_dir / "final_trajectory.jsonl"
        if path.exists():
            self.final_trajectory = _load_jsonl(path)
        else:
            self.final_trajectory = pd.DataFrame()
        return self.final_trajectory

    def load_prediction_folder(
        self, prediction_index: int
    ) -> Optional[PredictionFolderData]:
        """Load artifacts from a per-prediction folder."""
        folder = self.trajectory_dir / "predictions" / f"prediction_{prediction_index:03d}"
        if not folder.is_dir():
            return None
        return _load_prediction_folder(folder, prediction_index)

    def list_prediction_folders(self) -> List[int]:
        """List available prediction folder indices."""
        pred_dir = self.trajectory_dir / "predictions"
        if not pred_dir.is_dir():
            return []
        indices = []
        for d in sorted(pred_dir.iterdir()):
            if d.is_dir() and d.name.startswith("prediction_"):
                try:
                    idx = int(d.name.split("_")[1])
                    indices.append(idx)
                except (ValueError, IndexError):
                    pass
        return indices


@dataclass
class ExperimentData:
    """All loaded data for a single experiment run."""

    experiment_dir: Path
    config: Dict[str, Any] = field(default_factory=dict)
    batch_summary: Optional[pd.DataFrame] = None
    experiment_summary: Dict[str, Any] = field(default_factory=dict)
    trajectory_labels: List[str] = field(default_factory=list)

    def load_trajectory(self, file_label: str) -> TrajectoryData:
        """Load data for a single trajectory."""
        traj_dir = self.experiment_dir / file_label
        summary = _load_json(traj_dir / "experiment_summary.json")
        has_folders = (traj_dir / "predictions").is_dir()
        return TrajectoryData(
            file_label=file_label,
            trajectory_dir=traj_dir,
            summary=summary,
            has_prediction_folders=has_folders,
        )


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------


def load_experiment(experiment_dir: str | Path) -> ExperimentData:
    """Load an experiment from a directory path."""
    exp_dir = Path(experiment_dir)

    config = _load_yaml(exp_dir / "run_config.yaml")
    batch_summary = _load_csv(exp_dir / "batch_summary.csv")
    experiment_summary = _load_json(exp_dir / "experiment_summary.json")

    # Discover trajectory labels from subdirectories that have experiment_summary.json
    labels = []
    for child in sorted(exp_dir.iterdir()):
        if child.is_dir() and (child / "experiment_summary.json").exists():
            labels.append(child.name)

    return ExperimentData(
        experiment_dir=exp_dir,
        config=config,
        batch_summary=batch_summary,
        experiment_summary=experiment_summary,
        trajectory_labels=labels,
    )


def discover_experiments(base_dir: str | Path, max_depth: int = 5) -> List[Tuple[str, Path]]:
    """Discover experiment directories by finding batch_summary.csv files.

    Searches recursively (up to *max_depth* levels deep) for directories
    containing ``batch_summary.csv``.  Hidden folders (names starting with
    ``.``) are skipped.

    Returns list of (display_name, experiment_dir) tuples sorted by name.
    The display_name uses the relative path from *base_dir* so experiments
    in nested directories are easy to distinguish.

    Note: when Streamlit is available, prefer :func:`cached_discover_experiments`
    to avoid repeated filesystem walks on every app rerun.
    """
    base = Path(base_dir)
    if not base.exists():
        return []

    experiments = []

    def _walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(directory.iterdir())
        except PermissionError:
            return
        for child in children:
            if not child.is_dir():
                continue
            # Skip hidden folders (e.g. .worker_caches, .git)
            if child.name.startswith("."):
                continue
            if (child / "batch_summary.csv").exists():
                try:
                    rel = child.relative_to(base)
                    display_name = str(rel).replace("\\", "/") if str(rel) != "." else base.name
                except ValueError:
                    display_name = child.name
                experiments.append((display_name, child))
            # Keep searching deeper even if this dir had a CSV (nested experiments)
            _walk(child, depth + 1)

    # Check if the base dir itself is an experiment
    if (base / "batch_summary.csv").exists():
        experiments.append((base.name, base))

    _walk(base, 1)

    return experiments


def load_predictions_jsonl(path: Path) -> pd.DataFrame:
    """Load predictions.jsonl into a DataFrame, flattening nested fields."""
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for record in _iter_jsonl(path):
        flat = _flatten_prediction_record(record)
        rows.append(flat)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_timeline_jsonl(path: Path) -> pd.DataFrame:
    """Load timeline.jsonl into a DataFrame."""
    return _load_jsonl(path)


# ---------------------------------------------------------------------------
# Operation categorization (shared across dashboard pages)
# ---------------------------------------------------------------------------


def categorize_op(op_type: str) -> str:
    """Map a symbolic operation name to a broad category."""
    if op_type == "INPUT":
        return "Input"
    if op_type == "PASTE_FROM":
        return "Paste"
    if op_type.startswith("BORDER"):
        return "Border"
    if op_type.startswith("FONT"):
        return "Font"
    if op_type == "FILL_COLOR":
        return "Fill"
    if op_type in ("MERGE", "UNMERGE"):
        return "Merge"
    if op_type.startswith("ALIGN") or op_type in ("WRAP_TEXT", "TEXT_ORIENTATION"):
        return "Alignment"
    if op_type == "NUMBER_FORMAT":
        return "Number Format"
    return "Other"


def op_type_from_symbolic(symbolic_op: str) -> str:
    """Extract the operation type name from a symbolic string like 'INPUT | Sheet1!A1 | ...'."""
    return symbolic_op.split("|")[0].strip() if "|" in symbolic_op else symbolic_op.strip()


def cell_count_from_symbolic(symbolic_op: str) -> int:
    """Count cells covered by the range of a symbolic operation.

    ``INPUT | Sheet1!A1 | ...``  → 1
    ``FONT_BOLD | Sheet1!A1:C3 | True``  → 9  (3 rows × 3 cols)

    Returns 1 if the range cannot be parsed.
    """
    import re

    parts = symbolic_op.split("|")
    if len(parts) < 2:
        return 1

    ref = parts[1].strip()
    # Strip optional sheet prefix: "Sheet1!A1:C3" → "A1:C3"
    if "!" in ref:
        ref = ref.split("!", 1)[1]

    # Match single cell or range
    m = re.match(r"^([A-Z]+)(\d+)(?::([A-Z]+)(\d+))?$", ref, re.IGNORECASE)
    if not m:
        return 1

    col1 = _col_to_num(m.group(1))
    row1 = int(m.group(2))

    if m.group(3) is None:
        return 1  # single cell

    col2 = _col_to_num(m.group(3))
    row2 = int(m.group(4))

    return max(1, (abs(row2 - row1) + 1) * (abs(col2 - col1) + 1))


def _col_to_num(col_str: str) -> int:
    """Convert Excel column letters to number: A→1, Z→26, AA→27."""
    n = 0
    for ch in col_str.upper():
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = list(_iter_jsonl(path))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _flatten_prediction_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested fields in a prediction JSONL record."""
    flat: Dict[str, Any] = {}

    # Top-level scalars
    for key in (
        "prediction_index",
        "step_t",
        "user_step",
        "generation_time_s",
        "predicted_count",
        "accepted",
    ):
        if key in record:
            flat[key] = record[key]

    # Tokens
    tokens = record.get("tokens", {})
    if isinstance(tokens, dict):
        flat["tokens_input"] = tokens.get("input", 0)
        flat["tokens_output"] = tokens.get("output", 0)
        flat["tokens_total"] = tokens.get("total", 0)

    # Eval metrics
    metrics = record.get("eval_metrics", {})
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            flat[f"eval_{k}"] = v

    # Heuristic
    heuristic = record.get("heuristic", {})
    if isinstance(heuristic, dict):
        flat["heuristic_name"] = heuristic.get("name", "")
        flat["heuristic_accepted"] = heuristic.get("accepted", None)

    # Lists (keep as-is for per-prediction inspection)
    flat["predicted_ops"] = record.get("predicted_ops", [])
    flat["gt_segment"] = record.get("gt_segment", [])

    # Undo preview (if present)
    undo = record.get("future_if_accepted")
    if undo and isinstance(undo, dict):
        flat["undo_net_gain"] = undo.get("net_gain", 0)
        flat["undo_dedup_gain"] = undo.get("dedup_gain", 0)
        flat["undo_inverse_cost"] = undo.get("inverse_cost", 0)

    return flat


def _load_prediction_folder(
    folder: Path, prediction_index: int
) -> PredictionFolderData:
    """Load all artifacts from a prediction folder."""
    data = PredictionFolderData(prediction_index=prediction_index)

    prompt_path = folder / "prompt.txt"
    if prompt_path.exists():
        data.prompt_text = prompt_path.read_text(encoding="utf-8")

    response_path = folder / "response.txt"
    if response_path.exists():
        data.response_text = response_path.read_text(encoding="utf-8")

    meta_path = folder / "response_meta.json"
    if meta_path.exists():
        data.response_meta = _load_json(meta_path)

    ops_path = folder / "predicted_ops.txt"
    if ops_path.exists():
        data.predicted_ops = [
            l for l in ops_path.read_text(encoding="utf-8").splitlines() if l.strip()
        ]

    gt_path = folder / "gt_segment.txt"
    if gt_path.exists():
        data.gt_segment = [
            l for l in gt_path.read_text(encoding="utf-8").splitlines() if l.strip()
        ]

    hist_path = folder / "history_context.txt"
    if hist_path.exists():
        data.history_context = [
            l for l in hist_path.read_text(encoding="utf-8").splitlines() if l.strip()
        ]

    eval_path = folder / "evaluation.json"
    if eval_path.exists():
        data.evaluation = _load_json(eval_path)

    accept_path = folder / "acceptance.json"
    if accept_path.exists():
        data.acceptance = _load_json(accept_path)

    future_path = folder / "future_edits.json"
    if future_path.exists():
        data.future_edits = _load_json(future_path)

    return data


# ---------------------------------------------------------------------------
# Tree builder for hierarchical experiment selection
# ---------------------------------------------------------------------------


def build_experiment_tree(
    experiments: List[Tuple[str, Path]],
) -> List[Tuple[str, List[Tuple[str, int]]]]:
    """Organise experiments into a hierarchical tree by folder structure.

    Groups experiments by their parent directory path relative to the base
    directory.  Root-level experiments (no parent folder) are grouped under
    ``(root)``.

    Args:
        experiments: List of ``(display_name, experiment_dir)`` from
                     :func:`discover_experiments`.

    Returns:
        Sorted list of ``(folder_display_name, [(leaf_name, original_index), ...])``.
    """
    folders: Dict[str, List[Tuple[str, int]]] = {}
    for idx, (display_name, _path) in enumerate(experiments):
        parts = display_name.replace("\\", "/").split("/")
        if len(parts) > 1:
            folder = "/".join(parts[:-1])
            leaf = parts[-1]
        else:
            folder = ""
            leaf = display_name
        folders.setdefault(folder, []).append((leaf, idx))

    result: List[Tuple[str, List[Tuple[str, int]]]] = []
    if "" in folders:
        result.append(("(root)", folders.pop("")))
    for folder in sorted(folders.keys()):
        result.append((folder, folders[folder]))
    return result


# ---------------------------------------------------------------------------
# Streamlit-cached versions
# ---------------------------------------------------------------------------

if _has_streamlit:

    @st.cache_data(ttl=300, show_spinner="Scanning for experiments…")
    def cached_discover_experiments(base_dir: str) -> List[Tuple[str, Path]]:
        """Discover experiments with Streamlit caching (avoids repeated rglob)."""
        return discover_experiments(base_dir)

    @st.cache_data(ttl=300)
    def cached_load_experiment(experiment_dir: str) -> ExperimentData:
        """Load experiment with Streamlit caching."""
        return load_experiment(experiment_dir)

    @st.cache_data(ttl=300)
    def cached_load_batch_csv(csv_path: str) -> pd.DataFrame:
        """Load batch_summary.csv with Streamlit caching."""
        df = _load_csv(Path(csv_path))
        return df if df is not None else pd.DataFrame()
