"""
Experiment Configuration Module

Typed config loading from YAML with sweep expansion.
Each YAML file defines one experiment configuration. The sweep section
generates multiple config variants via cross-product of parameter values.
"""

from __future__ import annotations

import copy
import glob as glob_mod
import itertools
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class SolverConfig:
    """Solver configuration."""

    type: str = "constant"
    """Solver type: 'constant', 'llm' (alias for 'chat'), 'chat', or 'completion'."""

    adapter: str = "openai"
    """LLM adapter: 'openai', 'local', or 'custom'."""

    model: str = "gpt-4o-mini"
    """Model name (ignored when adapter='custom')."""

    adapter_class: Optional[str] = None
    """Dotted import path to an LLMAdapter subclass. Required when adapter='custom'."""

    adapter_kwargs: Optional[Dict[str, Any]] = None
    """Keyword args forwarded to the custom adapter's constructor (adapter='custom' only)."""

    temperature: float = 0.0
    """Generation temperature."""

    max_tokens: int = 4096
    """Max tokens for LLM response."""

    num_op_to_pred: Optional[int] = None
    """Max operations the model should predict per call (None = no limit)."""

    cache_enabled: bool = True
    """Whether to enable LLM caching."""

    cache_path: Optional[str] = None
    """Override cache file path."""

    emit_intent: bool = False
    """Whether LLM should emit an intent line before operations."""

    remove_sheet_name: bool = True
    """Strip sheet names from context and re-add after prediction."""

    emit_stop_instruction: bool = False
    """Include 'Write STOP...' instruction in the completion prompt."""

    # --- Prompt configuration ---
    system_prompt: Optional[str] = None
    """Inline system prompt template (Jinja2). Overrides the built-in default (chat mode)."""

    system_prompt_file: Optional[str] = None
    """Path to a system prompt template file (Jinja2). Takes precedence over inline system_prompt."""

    user_prompt: Optional[str] = None
    """Inline user prompt template (Jinja2). Overrides the built-in default (chat mode)."""

    user_prompt_file: Optional[str] = None
    """Path to a user prompt template file (Jinja2). Takes precedence over inline user_prompt."""

    completion_prompt: Optional[str] = None
    """Inline completion prompt template (Jinja2). Used by CompletionSolver."""

    completion_prompt_file: Optional[str] = None
    """Path to a completion prompt template file (Jinja2). Used by CompletionSolver."""

    # --- Generation control ---
    stop_sequences: Optional[List[str]] = None
    """Stop sequences for LLM generation. None = solver default (adds 'STOP' for CompletionSolver)."""

    detect_repetition: bool = False
    """Enable post-hoc repetition detection on parsed operations."""

    max_cycle_len: int = 8
    """Max cycle length to scan for in repetition detection."""

    max_repeats: int = 3
    """Max allowed consecutive repetitions of a cycle before truncation."""

    confidence_threshold: Optional[float] = None
    """Min mean log-probability per operation. Operations below this are dropped."""

    repetition_penalty: Optional[float] = None
    """Repetition penalty for local models (passed to adapter)."""

    # --- Local model hardware settings ---
    device: Optional[str] = None
    """Device for local models ('cpu', 'cuda', 'cuda:0', 'auto'). None = adapter default."""

    torch_dtype: Optional[str] = None
    """Torch dtype for local models ('float16', 'bfloat16', 'float32', 'auto'). None = adapter default."""

    base_url: Optional[str] = None
    """Base URL override for OpenAI-compatible servers (e.g., vllm at http://localhost:8000/v1)."""

    # --- Non-LLM baseline settings ---
    training_data_path: Optional[str] = None
    """Path to JSONL training data (used by NGramSolver)."""

    max_n: int = 5
    """Max n-gram order (used by NGramSolver)."""

    num_predictions: int = 1
    """Number of predictions per call (used by NGramSolver v1 — deprecated)."""

    use_cell_patterns: bool = True
    """Use spatial cell patterns (used by NGramSolver v1 — deprecated)."""

    model_dir: Optional[str] = None
    """Path to model directory (used by XGBoostSolver, LSTMSolver)."""

    range_mode: str = "relative"
    """Range representation mode: 'absolute' or 'relative' (feature-based solvers)."""

    window_size: int = 10
    """Feature window size (used by XGBoostSolver)."""

    max_predictions: int = 5
    """Max predictions per call (used by feature-based solvers via DecodingConfig)."""

    stop_on_type_change: bool = True
    """Stop predicting when operation type changes (used via DecodingConfig)."""

    max_ngram_n: int = 5
    """Max n-gram order for online learning (used by OnlineNGramSolver)."""

    min_match_length: int = 2
    """Minimum suffix match length (used by OnlineNGramSolver)."""

    # --- Symbolic transforms ---
    transforms: Optional[List[Dict[str, Any]]] = None
    """List of symbolic transform configs to apply via TransformedSolver.
    Each dict must have a ``"type"`` key (e.g. ``"relative_range"``).
    ``None`` means no transforms (default, backward-compatible)."""


@dataclass
class StrideSpec:
    """Stride specification (maps to StrideConfig at runtime)."""

    mode: str = "every_step"
    """Stride mode: 'every_step' or 'fixed_interval'."""

    interval: int = 1
    """Interval for fixed_interval mode."""


@dataclass
class ExperimentConfig:
    """
    A single experiment configuration.

    One YAML file maps to one ExperimentConfig. After sweep expansion,
    each variant also gets a distinct ``variant_name``.

    Example YAML::

        name: baseline_openai
        trajectory_paths:
          - data/trajectories/*.json
        max_runs: 16
        workers: 4
        heuristics:
          - ideal_user
        solver:
          type: llm
          adapter: openai
          model: gpt-4o-mini
        sweep:
          max_context_ops: [25, 50, 100]
    """

    # Identity
    name: str = "default"
    variant_name: str = "default"

    # Data
    trajectory_paths: List[str] = field(
        default_factory=lambda: ["data/trajectories/*.json"]
    )
    max_runs: Optional[int] = None
    max_steps: Optional[int] = None
    max_steps_pct: Optional[float] = None
    """Dynamic step cap as fraction of initial length (e.g., 1.2 = 120%).
    Effective max_steps = min(max_steps, max_steps_pct * initial_len)."""

    # Execution
    workers: int = 1
    max_context_ops: Optional[int] = 50
    """Max operations in prediction context. ``None`` = pass full history
    (useful with transforms; inner solver handles its own truncation)."""
    online_mode: bool = False

    # Components
    solver: SolverConfig = field(default_factory=SolverConfig)
    stride: StrideSpec = field(default_factory=StrideSpec)
    heuristics: List[str] = field(default_factory=lambda: ["ideal_user"])

    # Context shortening
    context_shortening_enabled: bool = True
    """Shorten large values in history context sent to LLM."""
    context_shortening_max_chars: int = 128
    context_shortening_corner_cells_dim: int = 3
    """Rows/cols to keep from each corner of large 2D arrays."""
    context_shortening_max_cells_2d: Optional[int] = None
    """Max total cells before 2D truncation. None = auto (corner_cells_dim² × 4)."""

    # Prediction chaining
    repredict_after_accept: bool = False
    """When True, skip the user pop after an accepted prediction and let
    the solver predict again immediately (chained predictions)."""

    max_predictions_per_step: Optional[int] = None
    """Hard cap on predictions kept per solver call. When set, only the
    first *max_predictions_per_step* operations from each prediction are
    retained; the rest are discarded. Combine with
    ``repredict_after_accept=True`` to get greedy single-op streaks."""

    # Output
    output_dir: str = "outputs"
    save_prediction_folders: bool = True
    """Save per-prediction artifact folders for detailed inspection."""
    buffered_writes: bool = False
    """Buffer JSONL writes in memory and flush once at end. Faster on slow filesystems."""

    # Sweep overrides — each key maps to a list of values to cross-product.
    # Supports dot notation for nested fields (e.g., "solver.model").
    sweep: Optional[Dict[str, List[Any]]] = None

    # Sweep-zip overrides — like sweep but lists are zipped (paired), not
    # cross-producted.  All lists must have the same length.
    # Useful when two parameters must change in lockstep (e.g., model + cache).
    sweep_zip: Optional[Dict[str, List[Any]]] = None

    # Sweep-independent overrides — each parameter is swept one-at-a-time
    # while all others stay at their base values.  Total variants = sum of
    # all list lengths (not the product).
    sweep_independent: Optional[Dict[str, List[Any]]] = None

    def resolve_trajectories(self) -> List[Path]:
        """Expand glob patterns into concrete file paths, applying max_runs."""
        paths: List[str] = []
        for pattern in self.trajectory_paths:
            if "*" in pattern or "?" in pattern:
                paths.extend(sorted(glob_mod.glob(pattern)))
            else:
                paths.append(pattern)
        result = [Path(p) for p in paths]
        if self.max_runs is not None:
            result = result[: self.max_runs]
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


def _resolve_path(
    config_path: Path, rel_path: Optional[str]
) -> Optional[str]:
    """Resolve a relative file path against the config file's directory.

    Returns ``None`` when *rel_path* is ``None`` or empty.  Absolute paths
    are returned unchanged.  Relative paths are resolved against the parent
    directory of *config_path*.
    """
    if not rel_path:
        return None
    p = Path(rel_path)
    if p.is_absolute():
        return str(p)
    return str(config_path.parent / p)


def load_experiment_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    solver_data = data.pop("solver", {})
    stride_data = data.pop("stride", {})
    sweep_data = data.pop("sweep", None)
    sweep_zip_data = data.pop("sweep_zip", None)
    sweep_indep_data = data.pop("sweep_independent", None)
    cs_data = data.pop("context_shortening", {})

    solver_cfg = SolverConfig(
        type=solver_data.get("type", "constant"),
        adapter=solver_data.get("adapter", "openai"),
        model=solver_data.get("model", "gpt-4o-mini"),
        adapter_class=solver_data.get("adapter_class"),
        adapter_kwargs=solver_data.get("adapter_kwargs"),
        temperature=solver_data.get("temperature", 0.0),
        max_tokens=solver_data.get("max_tokens", 4096),
        num_op_to_pred=solver_data.get("num_op_to_pred"),
        cache_enabled=solver_data.get("cache_enabled", True),
        cache_path=solver_data.get("cache_path"),
        emit_intent=solver_data.get("emit_intent", False),
        remove_sheet_name=solver_data.get("remove_sheet_name", True),
        system_prompt=solver_data.get("system_prompt"),
        system_prompt_file=_resolve_path(config_path, solver_data.get("system_prompt_file")),
        user_prompt=solver_data.get("user_prompt"),
        user_prompt_file=_resolve_path(config_path, solver_data.get("user_prompt_file")),
        completion_prompt=solver_data.get("completion_prompt"),
        completion_prompt_file=_resolve_path(config_path, solver_data.get("completion_prompt_file")),
        stop_sequences=solver_data.get("stop_sequences"),
        detect_repetition=solver_data.get("detect_repetition", False),
        max_cycle_len=solver_data.get("max_cycle_len", 8),
        max_repeats=solver_data.get("max_repeats", 3),
        confidence_threshold=solver_data.get("confidence_threshold"),
        repetition_penalty=solver_data.get("repetition_penalty"),
        device=solver_data.get("device"),
        torch_dtype=solver_data.get("torch_dtype"),
        base_url=solver_data.get("base_url"),
    )

    stride_cfg = StrideSpec(
        mode=stride_data.get("mode", "every_step"),
        interval=stride_data.get("interval", 1),
    )

    config_name = data.get("name", config_path.stem)

    return ExperimentConfig(
        name=config_name,
        variant_name=config_name,
        trajectory_paths=data.get(
            "trajectory_paths",
            data.get("trajectories", ["data/trajectories/*.json"]),
        ),
        max_runs=data.get("max_runs"),
        max_steps=data.get("max_steps"),
        max_steps_pct=data.get("max_steps_pct"),
        workers=data.get("workers", 1),
        max_context_ops=data.get("max_context_ops", 50),
        online_mode=data.get("online_mode", False),
        solver=solver_cfg,
        stride=stride_cfg,
        heuristics=data.get("heuristics", ["ideal_user"]),
        context_shortening_enabled=cs_data.get("enabled", True),
        context_shortening_max_chars=cs_data.get("max_chars", 128),
        context_shortening_corner_cells_dim=cs_data.get("corner_cells_dim", 3),
        context_shortening_max_cells_2d=cs_data.get("max_cells_2d", None),
        repredict_after_accept=data.get("repredict_after_accept", False),
        max_predictions_per_step=data.get("max_predictions_per_step"),
        output_dir=data.get("output_dir", "outputs"),
        save_prediction_folders=data.get("save_prediction_folders", True),
        buffered_writes=data.get("buffered_writes", False),
        sweep=sweep_data,
        sweep_zip=sweep_zip_data,
        sweep_independent=sweep_indep_data,
    )


def _set_nested_attr(obj: Any, dotted_key: str, value: Any) -> None:
    """Set a nested attribute using dot notation (e.g., 'solver.model')."""
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def expand_sweep(base: ExperimentConfig) -> List[ExperimentConfig]:
    """
    Expand sweep overrides into a list of concrete configs.

    If no sweep/sweep_zip/sweep_independent section, returns ``[base]``.

    ``sweep`` cross-products all parameter values:

        # sweep: {max_context_ops: [25, 50], solver.model: [model-a, model-b]}
        # produces 4 variants (2 x 2)

    ``sweep_zip`` zips parameter lists in lockstep (all lists must be equal
    length):

        # sweep_zip:
        #   solver.model: [model-a, model-b]
        #   solver.cache_path: [caches/a.json, caches/b.json]
        # produces 2 variants (paired)

    ``sweep_independent`` sweeps each parameter one-at-a-time while keeping
    all others at their base values (total = sum of list lengths):

        # sweep_independent:
        #   stride.interval: [1, 4, 8]
        #   max_context_ops: [8, 32]
        # produces 5 variants (3 + 2)

    When both ``sweep`` and ``sweep_zip`` are present the cross-product of
    ``sweep`` is combined with each zipped row from ``sweep_zip``
    (total = product_size x zip_size).

    ``sweep_independent`` variants are appended after any sweep/sweep_zip
    variants.
    """
    has_sweep = bool(base.sweep)
    has_zip = bool(base.sweep_zip)
    has_indep = bool(base.sweep_independent)

    if not has_sweep and not has_zip and not has_indep:
        return [base]

    variants: List[ExperimentConfig] = []

    # --- cross-product combos from sweep (x sweep_zip) ---
    if has_sweep or has_zip:
        if has_sweep:
            cp_keys = sorted(base.sweep.keys())
            cp_combos = list(itertools.product(*(base.sweep[k] for k in cp_keys)))
        else:
            cp_keys: List[str] = []
            cp_combos = [()]

        if has_zip:
            zip_keys = sorted(base.sweep_zip.keys())
            zip_lists = [base.sweep_zip[k] for k in zip_keys]
            lengths = {len(v) for v in zip_lists}
            if len(lengths) != 1:
                raise ValueError(
                    f"All sweep_zip lists must have the same length, "
                    f"got lengths {sorted(lengths)}"
                )
            zip_combos = list(zip(*zip_lists))
        else:
            zip_keys: List[str] = []
            zip_combos = [()]

        for cp_vals in cp_combos:
            for zip_vals in zip_combos:
                cfg = copy.deepcopy(base)
                cfg.sweep = None
                cfg.sweep_zip = None
                cfg.sweep_independent = None

                name_parts = []
                for key, val in zip(cp_keys, cp_vals):
                    _set_nested_attr(cfg, key, val)
                    short_key = key.split(".")[-1]
                    name_parts.append(f"{short_key}={val}")
                for key, val in zip(zip_keys, zip_vals):
                    _set_nested_attr(cfg, key, val)
                    short_key = key.split(".")[-1]
                    # Skip noisy auxiliary keys from variant name
                    if "cache" not in short_key:
                        name_parts.append(f"{short_key}={val}")

                cfg.variant_name = "_".join(name_parts)
                variants.append(cfg)

    # --- independent (one-at-a-time) combos from sweep_independent ---
    if has_indep:
        for key in sorted(base.sweep_independent.keys()):
            short_key = key.split(".")[-1]
            for val in base.sweep_independent[key]:
                cfg = copy.deepcopy(base)
                cfg.sweep = None
                cfg.sweep_zip = None
                cfg.sweep_independent = None

                _set_nested_attr(cfg, key, val)
                cfg.variant_name = f"{short_key}={val}"
                variants.append(cfg)

    return variants


__all__ = [
    "SolverConfig",
    "StrideSpec",
    "ExperimentConfig",
    "load_experiment_config",
    "expand_sweep",
]
