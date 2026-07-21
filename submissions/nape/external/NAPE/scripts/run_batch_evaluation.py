#!/usr/bin/env python
"""
Batch Evaluation Runner

Runs evaluation experiments across multiple trajectory files in parallel.
Supports auto-resume, incremental CSV checkpointing, and tqdm progress bars.

Usage:
    # Run with config
    python scripts/run_batch_evaluation.py --config configs/evaluation/batch.yaml

    # Run with command line args
    python scripts/run_batch_evaluation.py \\
        --trajectories data/trajectories/*.json \\
        --workers 4 --max-runs 16

    # Force re-run (no resume)
    python scripts/run_batch_evaluation.py --config configs/evaluation/batch.yaml --no-resume
"""

import argparse
import json
import logging
import os
import signal
import shutil
import sys
import time
import threading
import traceback as tb_mod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Global flag for graceful shutdown on Ctrl+C
_shutdown_event = threading.Event()
_ctrl_c_count = 0


def _signal_handler(_signum, _frame):
    global _ctrl_c_count
    _ctrl_c_count += 1
    if _ctrl_c_count >= 2:
        print("\nForce exit!")
        os._exit(1)
    _shutdown_event.set()
    print("\nCtrl+C — finishing current work, press again to force exit...")


signal.signal(signal.SIGINT, _signal_handler)

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from next_action_pred_eval.evaluation import (
    Orchestrator,
    every_step,
    every_n_steps,
    get_heuristic_by_name,
    HEURISTIC_IDEAL_USER,
)
from next_action_pred_eval.evaluation.solver import ConstantSolver
from next_action_pred_eval.evaluation.future_edits import FutureEditsConfig
from next_action_pred_eval.core.symbolic import symbolic_to_operations
from next_action_pred_eval.evaluation.experiment_config import (
    ExperimentConfig,
    load_experiment_config,
)
from next_action_pred_eval.evaluation.output_layout import (
    OutputLayout,
    TrajectoryResult,
)

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable


class TeeLogger:
    """Duplicates stdout/stderr to both terminal and a log file."""

    def __init__(self, log_file_path: Path):
        self.log_file_path = log_file_path
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_file_path, "w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self._closed = False

    def write(self, message: str):
        try:
            self.stdout.write(message)
            self.stdout.flush()
        except Exception:
            pass
        if not self._closed:
            try:
                self.log_file.write(message)
                self.log_file.flush()
            except (ValueError, OSError):
                pass

    def flush(self):
        try:
            self.stdout.flush()
        except Exception:
            pass
        if not self._closed:
            try:
                self.log_file.flush()
            except (ValueError, OSError):
                pass

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self._closed = True
        try:
            self.log_file.close()
        except Exception:
            pass
        return False


@dataclass
class WorkItem:
    """A single unit of work: one trajectory."""
    trajectory_path: Path
    file_label: str
    config_variant: str


def load_trajectory(path: Path) -> List[str]:
    """Load trajectory operations from file."""
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("operations", [])
    elif path.suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        raise ValueError(f"Unsupported trajectory format: {path.suffix}")


def create_solver(config: ExperimentConfig, cache_path_override: Optional[str] = None):
    """Create solver based on configuration."""
    solver_cfg = config.solver
    solver: ISolver

    if solver_cfg.type == "constant":
        solver = ConstantSolver()

    elif solver_cfg.type in ("llm", "chat", "completion"):
        from next_action_pred_eval.evaluation.baselines import ChatSolver, CompletionSolver
        from next_action_pred_eval.utils.llm import create_adapter

        if solver_cfg.adapter == "openai":
            kwargs = {"model": solver_cfg.model}
            if solver_cfg.base_url:
                kwargs["base_url"] = solver_cfg.base_url
                kwargs["api_key"] = "not-needed"
            adapter = create_adapter("openai", **kwargs)
        elif solver_cfg.adapter == "local":
            kwargs = {
                "model": solver_cfg.model,
                "cache_enabled": solver_cfg.cache_enabled,
                "cache_path": cache_path_override or solver_cfg.cache_path,
            }
            if solver_cfg.device:
                kwargs["device"] = solver_cfg.device
            if solver_cfg.torch_dtype:
                kwargs["torch_dtype"] = solver_cfg.torch_dtype
            adapter = create_adapter("local", **kwargs)
        elif solver_cfg.adapter == "custom":
            adapter = create_adapter(
                "custom",
                adapter_class=solver_cfg.adapter_class,
                adapter_kwargs=solver_cfg.adapter_kwargs or {},
            )
        else:
            raise ValueError(
                f"Unknown adapter type: {solver_cfg.adapter!r}. "
                "Expected one of: openai, local, custom."
            )

        # Select solver class based on type
        SolverClass = CompletionSolver if solver_cfg.type == "completion" else ChatSolver

        solver = SolverClass(
            llm_adapter=adapter,
            max_context_ops=config.max_context_ops,
            max_tokens=solver_cfg.max_tokens,
            temperature=solver_cfg.temperature,
            num_op_to_pred=solver_cfg.num_op_to_pred,
            enable_context_shortening=config.context_shortening_enabled,
            context_shortening_max_chars=config.context_shortening_max_chars,
            context_shortening_corner_cells_dim=config.context_shortening_corner_cells_dim,
            context_shortening_max_cells_2d=config.context_shortening_max_cells_2d,
            emit_intent=solver_cfg.emit_intent,
            remove_sheet_name=solver_cfg.remove_sheet_name,
            emit_stop_instruction=solver_cfg.emit_stop_instruction,
            stop_sequences=solver_cfg.stop_sequences,
            detect_repetition=solver_cfg.detect_repetition,
            max_cycle_len=solver_cfg.max_cycle_len,
            max_repeats=solver_cfg.max_repeats,
            confidence_threshold=solver_cfg.confidence_threshold,
            repetition_penalty=solver_cfg.repetition_penalty,
            system_prompt=solver_cfg.system_prompt,
            system_prompt_file=solver_cfg.system_prompt_file,
            user_prompt=solver_cfg.user_prompt,
            user_prompt_file=solver_cfg.user_prompt_file,
            completion_prompt=solver_cfg.completion_prompt,
            completion_prompt_file=solver_cfg.completion_prompt_file,
        )

    elif solver_cfg.type == "ngram":
        from next_action_pred_eval.evaluation.baselines import NGramSolver
        from next_action_pred_eval.evaluation.solver import DecodingConfig
        solver = NGramSolver(
            training_data_path=solver_cfg.training_data_path,
            range_mode=solver_cfg.range_mode,
            max_n=solver_cfg.max_n,
            decoding=DecodingConfig(
                max_predictions=solver_cfg.max_predictions,
                stop_on_type_change=solver_cfg.stop_on_type_change,
            ),
        )
    elif solver_cfg.type == "xgboost":
        from next_action_pred_eval.evaluation.baselines import XGBoostSolver
        from next_action_pred_eval.evaluation.solver import DecodingConfig
        solver = XGBoostSolver(
            model_dir=solver_cfg.model_dir or "models/xgboost",
            range_mode=solver_cfg.range_mode,
            window_size=solver_cfg.window_size,
            decoding=DecodingConfig(
                max_predictions=solver_cfg.max_predictions,
                stop_on_type_change=solver_cfg.stop_on_type_change,
            ),
        )
    elif solver_cfg.type == "lstm":
        from next_action_pred_eval.evaluation.baselines import LSTMSolver
        from next_action_pred_eval.evaluation.solver import DecodingConfig
        solver = LSTMSolver(
            model_dir=solver_cfg.model_dir or "models/lstm",
            range_mode=solver_cfg.range_mode,
            device=solver_cfg.device or "cpu",
            decoding=DecodingConfig(
                max_predictions=solver_cfg.max_predictions,
                stop_on_type_change=solver_cfg.stop_on_type_change,
            ),
        )
    elif solver_cfg.type == "online_ngram":
        from next_action_pred_eval.evaluation.baselines import OnlineNGramSolver
        from next_action_pred_eval.evaluation.solver import DecodingConfig
        solver = OnlineNGramSolver(
            range_mode=solver_cfg.range_mode,
            max_ngram_n=solver_cfg.max_ngram_n,
            min_match_length=solver_cfg.min_match_length,
            decoding=DecodingConfig(
                max_predictions=solver_cfg.max_predictions,
                stop_on_type_change=solver_cfg.stop_on_type_change,
            ),
        )
    else:
        raise ValueError(f"Unknown solver type: {solver_cfg.type}")

    # Wrap with transforms if configured
    if solver_cfg.transforms:
        from next_action_pred_eval.core.transforms import build_transforms
        from next_action_pred_eval.evaluation.transformed_solver import TransformedSolver

        transforms = build_transforms(solver_cfg.transforms)
        solver = TransformedSolver(inner=solver, transforms=transforms)
        logger.debug(
            "Wrapped solver with TransformedSolver: %s",
            [t.get_config() for t in transforms],
        )

    return solver


def create_stride_config(config: ExperimentConfig):
    """Create stride configuration."""
    if config.stride.mode == "every_step":
        return every_step()
    elif config.stride.mode == "fixed_interval":
        return every_n_steps(config.stride.interval)
    else:
        return every_step()


def create_heuristics(config: ExperimentConfig):
    """Create acceptance heuristics."""
    heuristics = []
    for h_name in config.heuristics:
        h = get_heuristic_by_name(h_name)
        if h:
            heuristics.append(h)
    return heuristics if heuristics else [HEURISTIC_IDEAL_USER]


def build_work_items(
    config: ExperimentConfig,
    layout: OutputLayout,
    resume: bool = True,
) -> tuple[List[WorkItem], List[TrajectoryResult]]:
    """
    Build list of WorkItems from config.
    Returns (work_items_to_run, completed_results_from_resume).
    """
    trajectories = config.resolve_trajectories()
    items: List[WorkItem] = []
    completed: List[TrajectoryResult] = []
    skipped = 0

    for traj_path in trajectories:
        file_label = traj_path.stem

        if resume and layout.is_completed(file_label):
            skipped += 1
            result = layout.load_completed_result(file_label, config.variant_name)
            if result:
                completed.append(result)
            logger.info("Skipping completed: %s", file_label)
            continue

        items.append(WorkItem(
            trajectory_path=traj_path,
            file_label=file_label,
            config_variant=config.variant_name,
        ))

    if skipped:
        logger.info("Resume: %d completed, %d remaining", skipped, len(items))
    return items, completed


def run_single_work_item(
    item: WorkItem,
    config: ExperimentConfig,
    layout: OutputLayout,
    worker_cache_path: Optional[str] = None,
) -> TrajectoryResult:
    """Execute one trajectory work item."""
    start_time = time.time()

    try:
        # Load trajectory
        symbolic_ops = load_trajectory(item.trajectory_path)
        if not symbolic_ops:
            return TrajectoryResult(
                file_label=item.file_label,
                config_variant=item.config_variant,
                status="error",
                error_message="Empty trajectory",
            )

        operations = symbolic_to_operations(symbolic_ops)

        # Create components
        solver = create_solver(config, cache_path_override=worker_cache_path)
        stride_config = create_stride_config(config)
        heuristics = create_heuristics(config)

        # Create orchestrator — output goes to layout's run dir
        run_dir = layout.get_run_dir(item.file_label)
        orchestrator = Orchestrator(
            solver=solver,
            stride_config=stride_config,
            acceptance_heuristics=heuristics,
            output_dir=run_dir,
            future_edits_config=FutureEditsConfig() if config.online_mode else None,
            save_prediction_folders=config.save_prediction_folders,
            repredict_after_accept=config.repredict_after_accept,
            max_predictions_per_step=config.max_predictions_per_step,
            buffered_writes=config.buffered_writes,
        )

        # Compute effective max_steps: min(max_steps, max_steps_pct * initial_len)
        effective_max_steps = config.max_steps
        if config.max_steps_pct is not None:
            pct_limit = int(config.max_steps_pct * len(operations))
            if effective_max_steps is not None:
                effective_max_steps = min(effective_max_steps, pct_limit)
            else:
                effective_max_steps = pct_limit

        # Run experiment
        summary = orchestrator.run_experiment(
            action_stream=operations,
            experiment_name=item.file_label,
            max_context_ops=config.max_context_ops,
            online_mode=config.online_mode,
            max_steps=effective_max_steps,
        )

        run_time = time.time() - start_time

        # Read empty/errored counts from the per-trajectory summary written by the recorder
        traj_summary_path = run_dir / "experiment_summary.json"
        empty_preds = 0
        errored_empty_preds = 0
        if traj_summary_path.exists():
            try:
                with open(traj_summary_path, "r", encoding="utf-8") as f:
                    traj_summary_data = json.load(f)
                empty_preds = traj_summary_data.get("empty_predictions", 0)
                errored_empty_preds = traj_summary_data.get("errored_empty_predictions", 0)
            except (json.JSONDecodeError, OSError):
                pass

        # Map ExperimentSummary -> TrajectoryResult
        initial = summary.total_steps
        user_steps = summary.user_steps_taken
        attempted = summary.total_predictions
        accepted = summary.total_accepted

        coverage_total = summary.total_tp + summary.total_fp + summary.total_fn
        coverage_pct_tp = summary.total_tp / coverage_total if coverage_total > 0 else 0.0

        return TrajectoryResult(
            file_label=item.file_label,
            config_variant=item.config_variant,
            status="success",
            net_operations_saved=summary.total_ops_saved,
            predictions_attempted=attempted,
            predictions_accepted=accepted,
            empty_predictions=empty_preds,
            errored_empty_predictions=errored_empty_preds,
            initial_sequence_length=initial,
            final_sequence_length=summary.final_sequence_length,
            user_steps_taken=user_steps,
            uas_pct=summary.uas_pct,
            total_formatting_ops=summary.total_formatting_ops,
            ufas=summary.ufas,
            ufas_pct=summary.ufas_pct,
            acceptance_rate=summary.acceptance_rate,
            avg_precision=(
                sum(d.get("precision", 0) for d in summary.prediction_details)
                / len(summary.prediction_details)
                if summary.prediction_details else 0.0
            ),
            coverage_pct_tp=coverage_pct_tp,
            ops_saved_per_prediction=(
                summary.total_ops_saved / attempted if attempted > 0 else 0.0
            ),
            total_tokens=summary.total_tokens,
            input_tokens=summary.total_input_tokens,
            output_tokens=summary.total_output_tokens,
            total_time=run_time,
            inverse_ops_added=summary.total_inverse_ops_added,
            user_step_limit_reached=summary.metadata.get("user_step_limit_reached", False),
            per_heuristic_stats=summary.per_heuristic_stats or None,
        )

    except Exception as e:
        logger.error("Error processing %s: %s", item.file_label, e)
        tb_str = tb_mod.format_exc()
        logger.debug(tb_str)
        layout.append_failed_run(
            item.file_label, item.config_variant, str(e), tb_str,
        )
        return TrajectoryResult(
            file_label=item.file_label,
            config_variant=item.config_variant,
            status="error",
            error_message=str(e),
            total_time=time.time() - start_time,
        )


def _merge_worker_caches(
    main_cache_path: str,
    worker_cache_paths: List[Path],
    worker_caches_dir: Path,
) -> None:
    """Merge per-worker cache files back into the main cache."""
    main_path = Path(main_cache_path)
    main_path.parent.mkdir(parents=True, exist_ok=True)

    main_data = {}
    if main_path.exists():
        try:
            with open(main_path, "r", encoding="utf-8") as f:
                main_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Main cache %s corrupted, starting fresh", main_path)

    entries_added = 0
    merged = 0
    for wp in worker_cache_paths:
        if not wp.exists():
            continue
        try:
            with open(wp, "r", encoding="utf-8") as f:
                worker_data = json.load(f)
            before = len(main_data)
            main_data.update(worker_data)
            entries_added += len(main_data) - before
            merged += 1
        except json.JSONDecodeError:
            logger.warning("Worker cache corrupted, skipping: %s", wp.name)

    if entries_added > 0:
        with open(main_path, "w", encoding="utf-8") as f:
            json.dump(main_data, f, indent=2)
        logger.info("Merged %d worker caches -> %s (+%d entries)", merged, main_path.name, entries_added)

    try:
        shutil.rmtree(worker_caches_dir)
    except Exception as e:
        logger.warning("Could not remove worker caches dir: %s", e)


def run_experiment_config(
    config: ExperimentConfig,
    resume: bool = True,
) -> List[TrajectoryResult]:
    """
    Main entry point: runs all work items for one ExperimentConfig.

    This is what ``run_sweep.py`` calls for each variant.
    """
    batch_start = time.time()

    # Create output layout
    layout = OutputLayout(config.output_dir)

    # Save run config
    layout.save_run_config(config.to_dict())

    # Set up logging
    log_file_path = layout.base_dir / "run.log"

    with TeeLogger(log_file_path) as tee:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(tee)],
            force=True,
        )

        # Suppress noisy per-prediction warnings from terminal.
        # These still go to trajectory.log via ExperimentRecorder.
        for noisy_logger in [
            "next_action_pred_eval.core.symbolic",
            "next_action_pred_eval.utils.llm",
            "next_action_pred_eval.evaluation.baselines",
            "next_action_pred_eval.evaluation.future_edits",
        ]:
            logging.getLogger(noisy_logger).setLevel(logging.ERROR)

        # Print configuration
        print("=" * 60)
        print("BATCH EVALUATION")
        print("=" * 60)
        print(f"Config:       {config.variant_name}")
        print(f"Output:       {config.output_dir}")
        print(f"Solver:       {config.solver.type}", end="")
        if config.solver.type == "llm":
            print(f" ({config.solver.adapter}/{config.solver.model})")
        else:
            print()
        print(f"Heuristics:   {', '.join(config.heuristics)}")
        print(f"Workers:      {config.workers}")
        print(f"Online mode:  {config.online_mode}")
        print(f"Max steps:    {config.max_steps or 'all'}")
        print("=" * 60)
        print()

        # Build work items (with resume detection)
        work_items, completed_results = build_work_items(config, layout, resume=resume)
        results: List[TrajectoryResult] = list(completed_results)

        if not work_items:
            logger.info("All trajectories already completed.")
            # Still write final CSV with completed results
            if results:
                layout.write_csv(results)
                layout.write_experiment_summary(results, config.to_dict(), time.time() - batch_start)
            return results

        trajectories_str = f"{len(work_items)} to run"
        if completed_results:
            trajectories_str += f" ({len(completed_results)} resumed)"
        print(f"Trajectories: {trajectories_str}")
        print()

        interrupted = False

        try:
            if config.workers == 1:
                # Sequential execution with tqdm
                for item in tqdm(work_items, desc="Running", unit="traj"):
                    if _shutdown_event.is_set():
                        interrupted = True
                        break
                    result = run_single_work_item(item, config, layout)
                    results.append(result)
                    layout.append_csv_row(result)
            else:
                # Parallel execution
                main_cache_path = config.solver.cache_path or "caches/llm_cache.json"
                worker_caches_dir = layout.base_dir / ".worker_caches"
                worker_caches_dir.mkdir(parents=True, exist_ok=True)
                worker_cache_paths: List[Path] = []

                executor = ThreadPoolExecutor(max_workers=config.workers)
                future_to_item = {}
                for i, item in enumerate(work_items):
                    worker_id = f"w{i}_{item.file_label}"
                    worker_cache = worker_caches_dir / f"cache_{worker_id}.json"
                    worker_cache_paths.append(worker_cache)
                    future_to_item[executor.submit(
                        run_single_work_item,
                        item, config, layout,
                        worker_cache_path=str(worker_cache),
                    )] = item

                ok_count = 0
                fail_count = 0
                total_futures = len(future_to_item)
                collected = 0
                pbar = tqdm(total=total_futures, desc="Running", unit="traj")
                try:
                    while collected < total_futures:
                        # time.sleep IS interruptible by Ctrl+C on Windows
                        time.sleep(0.5)
                        # Check for newly completed futures
                        newly_done = [f for f in future_to_item if f.done() and f not in getattr(pbar, '_seen', set())]
                        if not hasattr(pbar, '_seen'):
                            pbar._seen = set()
                        for future in newly_done:
                            pbar._seen.add(future)
                            collected += 1
                            item = future_to_item[future]
                            try:
                                result = future.result(timeout=0)
                            except Exception as e:
                                result = TrajectoryResult(
                                    file_label=item.file_label,
                                    config_variant=item.config_variant,
                                    status="error",
                                    error_message=str(e),
                                )
                                layout.append_failed_run(
                                    item.file_label, item.config_variant,
                                    str(e), tb_mod.format_exc(),
                                )
                            results.append(result)
                            layout.append_csv_row(result)

                            if result.status == "success":
                                ok_count += 1
                            else:
                                fail_count += 1
                            pbar.update(1)
                            pbar.set_postfix(ok=ok_count, fail=fail_count)
                finally:
                    pbar.close()
                    executor.shutdown(wait=False, cancel_futures=True)

                # Merge worker caches
                _merge_worker_caches(main_cache_path, worker_cache_paths, worker_caches_dir)

        except KeyboardInterrupt:
            _shutdown_event.set()
            interrupted = True
            print()
            print("INTERRUPTED — saving partial results...")

        # Write final summaries (overwrite incremental CSV with complete data)
        wall_time = time.time() - batch_start
        layout.write_csv(results)
        layout.write_experiment_summary(results, config.to_dict(), wall_time)

        # Print summary
        successful = [r for r in results if r.status == "success"]
        failed = [r for r in results if r.status == "error"]
        print()
        print("=" * 60)
        print("INTERRUPTED — partial results saved" if interrupted else "BATCH COMPLETE")
        print("=" * 60)
        print(f"Successful:        {len(successful)}/{len(results)}")
        if successful:
            print(f"Total predictions: {sum(r.predictions_attempted for r in successful)}")
            print(f"Total accepted:    {sum(r.predictions_accepted for r in successful)}")
            print(f"Total ops saved:   {sum(r.net_operations_saved for r in successful)}")
            mean_uas = sum(r.uas_pct for r in successful) / len(successful)
            print(f"Mean UAS:          {mean_uas:.2%}")

            # Per-heuristic summary (offline multi-heuristic evaluation)
            per_h_agg: Dict[str, Dict[str, int]] = {}
            total_initial_len = sum(r.initial_sequence_length for r in successful)
            for r in successful:
                if r.per_heuristic_stats:
                    for h_name, h_stats in r.per_heuristic_stats.items():
                        if h_name not in per_h_agg:
                            per_h_agg[h_name] = {"accepted": 0, "total": 0, "ops_saved": 0}
                        per_h_agg[h_name]["accepted"] += h_stats.get("accepted", 0)
                        per_h_agg[h_name]["total"] += h_stats.get("total_predictions", 0)
                        per_h_agg[h_name]["ops_saved"] += h_stats.get("total_ops_saved", 0)
            if per_h_agg:
                print()
                print("Per-heuristic stats (offline):")
                print(f"  {'Heuristic':25s} {'Accept Rate':>12s} {'UAS':>8s}  {'Ops Saved':>10s}")
                print(f"  {'-'*25} {'-'*12} {'-'*8}  {'-'*10}")
                for h_name, agg in per_h_agg.items():
                    rate = agg["accepted"] / agg["total"] if agg["total"] > 0 else 0.0
                    uas = agg["ops_saved"] / total_initial_len if total_initial_len > 0 else 0.0
                    print(f"  {h_name:25s} {rate:>11.2%} {uas:>8.2%}  {agg['ops_saved']:>10d}")
        if failed:
            print(f"Failed:            {len(failed)}")
        print(f"Wall time:         {wall_time:.1f}s")
        print(f"Results:           {layout.base_dir}")
        print("=" * 60)

        if not successful and results:
            print()
            print("WARNING: No successful runs!")
            print("Check the log file for errors:", log_file_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run batch evaluation experiments")
    parser.add_argument("--config", type=Path, help="Path to experiment config YAML")
    parser.add_argument("--trajectories", nargs="+", help="Trajectory files or glob patterns")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--max-runs", type=int, help="Max trajectories to process")
    parser.add_argument("--max-steps", type=int, help="Max steps per trajectory")
    parser.add_argument("--solver", choices=["constant", "llm"], default="constant")
    parser.add_argument("--adapter", choices=["openai", "local", "custom"], default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--heuristics", nargs="+", default=["ideal_user"])
    parser.add_argument("--online-mode", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Force re-run, don't skip completed")
    parser.add_argument(
        "--set", nargs="+", metavar="KEY=VALUE",
        help="Override config values using dot notation, e.g. --set solver.model=gpt-4o-mini output_dir=foo",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.config:
        config = load_experiment_config(args.config)
        # Apply --set overrides on top of the loaded config
        if args.set:
            from next_action_pred_eval.evaluation.experiment_config import _set_nested_attr
            for kv in args.set:
                if "=" not in kv:
                    parser.error(f"--set values must be KEY=VALUE, got: {kv}")
                key, val = kv.split("=", 1)
                # Auto-cast common types
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
                elif val.lower() == "null" or val.lower() == "none":
                    val = None
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass  # keep as string
                _set_nested_attr(config, key, val)
    else:
        # Build config from CLI args
        import glob
        trajectory_paths = []
        if args.trajectories:
            for pattern in args.trajectories:
                if "*" in pattern:
                    trajectory_paths.extend(glob.glob(pattern))
                else:
                    trajectory_paths.append(pattern)
        if not trajectory_paths:
            parser.error("Either --config or --trajectories must be provided")

        from next_action_pred_eval.evaluation.experiment_config import SolverConfig, StrideSpec
        config = ExperimentConfig(
            name="cli_run",
            variant_name="cli_run",
            trajectory_paths=trajectory_paths,
            max_runs=args.max_runs,
            max_steps=args.max_steps,
            workers=args.workers,
            solver=SolverConfig(type=args.solver, adapter=args.adapter, model=args.model),
            stride=StrideSpec(mode="every_step"),
            heuristics=args.heuristics,
            output_dir=str(args.output_dir),
            online_mode=args.online_mode,
        )

    results = run_experiment_config(config, resume=not args.no_resume)

    failed = sum(1 for r in results if r.status != "success")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
