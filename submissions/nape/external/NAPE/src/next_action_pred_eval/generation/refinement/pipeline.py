from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from next_action_pred_eval.core.operations import OPERATION_DOCS, OPERATION_MAP
from next_action_pred_eval.core.symbolic import operations_to_symbolic, symbolic_to_operations
from next_action_pred_eval.core.state import StateBuilder
from next_action_pred_eval.evaluation.state_comparator import ComparisonResult, StateComparator

from .config import RefinementConfig
from .image_capture import ensure_sheet_image
from .input_normalizer import (
    compress_operations,
    decompress_operations,
    get_compression_docs,
    is_compressed_input,
    CompressionStats,
)
from .llm_adapter import (
    LLMResponse,
    LLMServiceUnavailableError,
    Message,
    RefinementLLMAdapter,
    Role,
    TokenUsage,
)
from .loaders import filter_operations_by_scope, load_symbolic_operations
from .prompting import (
    build_feedback_message,
    build_judge_prompt,
    build_judge_system_prompt,
    build_original_instruction,
    build_retry_feedback_message,
    build_retry_mismatch_message,
    build_system_prompt,
    format_operation_catalog,
    format_operations_block,
    summarize_reference_operations,
    wrap_judge_feedback,
)
from .reporting import create_run_directory, save_iteration_artifacts, save_llm_call_log, write_final_report
from .utils import (
    JudgeVerdict,
    ParsedLLMOutput,
    SkipDeclaration,
    build_completion_moves,
    build_mismatch_operations,
    build_mismatch_report,
    build_retry_hint,
    filter_differences_by_skips,
    filter_to_prime_visible,
    list_mismatched_cells,
    parse_judge_response,
    parse_llm_response,
    summarize_comparison,
    summarize_extra_properties,
    summarize_sequence_diff,
)
from .workbook_utils import build_state_from_operations, filter_state

logger = logging.getLogger(__name__)


def _validate_symbolic_operations(operations: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Validate a list of symbolic operations and return valid/invalid/skipped ones.

    Handles #SKIP prefix for operations marked as skippable by LLM.
    Skipped operations are validated for syntax but tracked separately.

    This is a local implementation that replaces the reference repo's
    excel_to_steps_gen.operations_utils.validate_symbolic_operations.

    Args:
        operations: List of symbolic operation strings

    Returns:
        Tuple[List[str], List[str], List[str]]: (valid_operations, invalid_operations, skipped_operations)
    """
    import re

    valid_ops: List[str] = []
    invalid_ops: List[str] = []
    skipped_ops: List[str] = []

    valid_operation_types = set(OPERATION_MAP.keys())
    excel_range_pattern = re.compile(r'^[A-Za-z]{1,3}\d{1,7}(?::[A-Za-z]{1,3}\d{1,7})?$')

    for op in operations:
        try:
            # Check for #SKIP prefix
            is_skipped = False
            op_to_validate = op.strip()
            if op_to_validate.startswith('#SKIP '):
                is_skipped = True
                op_to_validate = op_to_validate[6:].strip()

            parts = op_to_validate.split(' | ')
            if len(parts) >= 2:
                operation_type = parts[0].strip()
                range_part = parts[1].strip()

                # Check if operation type is valid
                if operation_type in valid_operation_types:
                    # Basic range validation
                    range_to_check = range_part.split('!')[-1] if '!' in range_part else range_part
                    if excel_range_pattern.match(range_to_check):
                        if is_skipped:
                            skipped_ops.append(op)
                        else:
                            valid_ops.append(op)
                    else:
                        invalid_ops.append(op)
                else:
                    invalid_ops.append(op)
            else:
                invalid_ops.append(op)
        except Exception:
            invalid_ops.append(op)

    return valid_ops, invalid_ops, skipped_ops


@dataclass
class RefinementOutcome:
    success: bool
    iterations: int
    best_operations: List[str]
    run_directory: Optional[str]
    message: str


class SequenceRefinementPipeline:
    def __init__(
        self,
        config: RefinementConfig,
        llm_adapter: Optional[RefinementLLMAdapter] = None,
        judge_adapter: Optional[RefinementLLMAdapter] = None,
    ) -> None:
        self.config = config.resolve()
        if llm_adapter is None:
            raise ValueError(
                "llm_adapter is required. Create a RefinementLLMAdapter wrapping "
                "your LLMAdapter instance."
            )
        self.llm = llm_adapter
        self.judge = judge_adapter or llm_adapter
        self.comparator = StateComparator(ignore_defaults=True)
        self.llm_call_log: List[Dict[str, Any]] = []
        self.call_log_dir: Optional[Path] = None
        self.call_log_index: int = 0

    def run(self) -> RefinementOutcome:
        original_ops, skipped_lines = load_symbolic_operations(
            self.config.step_file, self.config.sheet_name, self.config.max_dimension
        )
        if not original_ops:
            return RefinementOutcome(False, 0, [], None, "No operations available after filtering")

        # Build target state from original operations instead of reading from Excel
        target_state = build_state_from_operations(
            original_ops,
            self.config.sheet_name,
            self.config.max_dimension,
            include_formatting=self.config.compare_formatting,
        )

        # Compress large INPUT operations for prompt efficiency
        compressed_ops: List[str] = original_ops
        compression_stats: Optional[CompressionStats] = None
        compressed_input_docs: str = ""
        if self.config.compress_inputs:
            compressed_ops, compression_stats = compress_operations(
                original_ops,
                cell_threshold=self.config.input_compression_threshold,
                string_threshold=self.config.input_string_truncate_threshold,
            )
            if compression_stats.compressed_operations > 0:
                compressed_input_docs = get_compression_docs()
                self._log_progress(
                    f"Compressed {compression_stats.compressed_operations}/{compression_stats.total_operations} "
                    f"INPUT operations ({compression_stats.savings_percent:.1f}% token reduction)"
                )

        system_prompt = build_system_prompt()
        operation_catalog = format_operation_catalog(OPERATION_DOCS)
        original_instruction = build_original_instruction(
            sheet_name=self.config.sheet_name,
            max_dimension=self.config.max_dimension,
            operation_catalog=operation_catalog,
            reference_ops_snippet=summarize_reference_operations(
                compressed_ops, self.config.reference_operation_limit
            ),
            compressed_input_docs=compressed_input_docs,
        )
        sheet_image = ensure_sheet_image(
            self.config.final_workbook,
            self.config.sheet_name,
            self.config.sheet_image_path,
            self.config.capture_dir,
            self.config.max_dimension,
            self.config.allow_image_capture,
        )

        run_dir = create_run_directory(self.config.output_dir)
        (run_dir / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")
        self.call_log_dir = run_dir / "calls"
        self.call_log_dir.mkdir(parents=True, exist_ok=True)
        self.llm_call_log.clear()
        self.call_log_index = 0

        best_candidate: Optional[Tuple[List[str], ComparisonResult, ParsedLLMOutput]] = None
        best_score: Optional[int] = None
        last_response_text: Optional[str] = None
        last_feedback_text: Optional[str] = None

        best_comparator_report: Optional[str] = None
        best_mismatched_cells: List[str] = []
        iterations_run = 0
        history_summaries: List[str] = []
        service_down_reason: Optional[str] = None
        judge_history: List[Dict[str, Any]] = []
        judge_rejection_reason: Optional[str] = None
        overall_usage = {"prompt": 0, "completion": 0, "total": 0}
        overall_elapsed = 0.0

        baseline_diff_summary = summarize_sequence_diff(compressed_ops, compressed_ops, limit=None)
        baseline_verdict, baseline_usage, baseline_elapsed = self._run_judge(
            compressed_ops,
            "Original heuristic sequence",
            "baseline",
            None,
            sheet_image,
            baseline_diff_summary,
        )
        for key in overall_usage:
            overall_usage[key] += baseline_usage[key]
        overall_elapsed += baseline_elapsed
        judge_history.append(self._judge_record(baseline_verdict, iteration_label="baseline"))
        last_feedback_text = wrap_judge_feedback(baseline_verdict.raw_text)
        self._log_progress(
            "Baseline judge usage — prompt: {prompt}, completion: {completion}, total: {total}, elapsed: {elapsed:.2f}s".format(
                prompt=baseline_usage["prompt"],
                completion=baseline_usage["completion"],
                total=baseline_usage["total"],
                elapsed=baseline_elapsed,
            )
        )
        self._update_history(
            history_summaries,
            f"Baseline judge: {'\u2705' if baseline_verdict.is_human else '\u26a0\ufe0f'} {baseline_verdict.rationale[:80]}",
        )
        last_response_text: Optional[str] = None
        accepted_candidate: Optional[List[str]] = None

        if baseline_verdict.is_human:
            summary = {
                "success": True,
                "iterations": 0,
                "best_score": 0,
                "sheet": self.config.sheet_name,
                "max_dimension": self.config.max_dimension,
                "mismatched_cells": [],
                "overall_token_usage": overall_usage,
                "overall_elapsed_seconds": overall_elapsed,
                "baseline_accepted": True,
                "accepted_sequence_source": "baseline",
                "judge_history": judge_history,
            }
            best_ops_list = original_ops.copy()
            best_ops_text = "\n".join(best_ops_list)
            write_final_report(run_dir, summary, None, best_ops_text)
            save_llm_call_log(run_dir, self.llm_call_log)
            return RefinementOutcome(
                True,
                0,
                best_ops_list,
                str(run_dir),
                "Baseline sequence already reads as human; no refinement needed.",
            )

        for iteration in range(1, self.config.max_iterations + 1):
            iterations_run = iteration
            history_context = self._format_history_summary(history_summaries)
            self._log_progress(f"Iteration {iteration}/{self.config.max_iterations} started")
            retry_response_text: Optional[str] = None
            retry_feedback_text: Optional[str] = None
            retry_history: List[Dict[str, Any]] = []
            iteration_usage = {"prompt": 0, "completion": 0, "total": 0}
            iteration_elapsed = 0.0

            parsed: Optional[ParsedLLMOutput] = None
            response_text: str = ""
            comparison: Optional[ComparisonResult] = None
            comparison_report: Optional[str] = None
            mismatched_cells: List[str] = []
            valid_ops: List[str] = []
            invalid_ops: List[str] = []
            skipped_ops: List[str] = []
            rejected: List[str] = []
            predicted_filtered_state: Optional[Dict[str, Any]] = None
            completion_moves: List[str] = []
            incorrect_operations: List[str] = []
            inner_success = False
            candidate_ops: Optional[List[str]] = None
            candidate_parsed: Optional[ParsedLLMOutput] = None
            candidate_comparison: Optional[ComparisonResult] = None
            candidate_report: Optional[str] = None
            best_attempt: Optional[Tuple[List[str], ComparisonResult, ParsedLLMOutput, str, List[str], Optional[str], List[str]]] = None
            best_attempt_score: Optional[int] = None

            for retry in range(1, self.config.max_retries + 1):
                self._log_progress(
                    f"Iteration {iteration}/{self.config.max_iterations} - "
                    f"retry {retry}/{self.config.max_retries}"
                )
                messages = self._build_messages(
                    system_prompt,
                    original_instruction,
                    sheet_image,
                    last_response_text,
                    last_feedback_text,
                    retry_response_text,
                    retry_feedback_text,
                    history_context,
                )
                request_overrides = self._build_request_overrides(retry)
                attempt_start = time.perf_counter()
                try:
                    response = self._complete_with_service_check(self.llm, messages, request_overrides)
                except LLMServiceUnavailableError as exc:
                    service_down_reason = (
                        "LLM provider returned a non-JSON outage page; aborting after"
                        f" {self.config.api_down_retry_attempts + 1} attempts."
                    )
                    logger.error("LLM service unavailable: %s", exc)
                    break
                attempt_elapsed = time.perf_counter() - attempt_start
                response_text = response.text or ""
                parsed = parse_llm_response(response_text, self.config.humaneness_keyword)
                usage_snapshot = self._extract_token_usage(response.usage)
                for key in iteration_usage:
                    iteration_usage[key] += usage_snapshot[key]
                    overall_usage[key] += usage_snapshot[key]
                iteration_elapsed += attempt_elapsed
                overall_elapsed += attempt_elapsed
                self._log_progress(
                    "Retry %s tokens — prompt: %s, completion: %s, total: %s, elapsed: %.2fs"
                    % (
                        retry,
                        usage_snapshot["prompt"],
                        usage_snapshot["completion"],
                        usage_snapshot["total"],
                        attempt_elapsed,
                    )
                )

                validation_errors = list(parsed.errors)
                scoped_ops, rejected = filter_operations_by_scope(
                    parsed.operations, self.config.sheet_name, self.config.max_dimension
                )

                # Decompress any compressed INPUT operations using the target state
                if self.config.compress_inputs:
                    scoped_ops = decompress_operations(
                        scoped_ops, target_state, self.config.sheet_name
                    )

                if rejected:
                    validation_errors.append(
                        f"Rejected {len(rejected)} operations outside the allowed sheet/bounds"
                    )

                valid_ops = []
                invalid_ops = []
                skipped_ops = []
                op_objects = []
                comparison = None
                comparison_report = None
                filtered_mismatches = []
                skipped_count = 0

                if not validation_errors:
                    valid_ops, invalid_ops, skipped_ops = _validate_symbolic_operations(scoped_ops)
                    if invalid_ops:
                        validation_errors.append(f"Invalid operations: {invalid_ops[:5]}")

                if not validation_errors and valid_ops:
                    try:
                        op_objects = symbolic_to_operations(valid_ops)
                    except Exception as exc:  # pragma: no cover - defensive
                        validation_errors.append(f"Failed to parse operations: {exc}")

                if not validation_errors and op_objects:
                    builder = StateBuilder()
                    predicted_state = builder.apply_operations(op_objects)
                    predicted_filtered = filter_state(
                        predicted_state,
                        self.config.sheet_name,
                        self.config.max_dimension,
                        include_formatting=self.config.compare_formatting,
                    )
                    predicted_filtered_state = predicted_filtered
                    comparison = self.comparator.compare(predicted_filtered, target_state)
                    mismatched_cells = list_mismatched_cells(comparison)
                    # Apply skip/ignore/correction filtering: remove differences the LLM explicitly declared
                    remaining_diffs, skipped_diffs = filter_differences_by_skips(
                        comparison.differences if comparison else [],
                        parsed.skip_declarations,
                        parsed.ignore_declarations,
                        parsed.correction_declarations,
                    )
                    filtered_mismatches = [d for d in remaining_diffs if d.match_type != "TP"]
                    skipped_count = len(skipped_diffs)
                    # Use filtered differences for mismatch report
                    comparison_report = build_mismatch_report(remaining_diffs)
                    # Compute completion moves from FN differences (filtered)
                    completion_moves = build_completion_moves(remaining_diffs)
                    # Compute incorrect operations from FP differences (filtered)
                    incorrect_operations = build_mismatch_operations(remaining_diffs)
                    # For validation score, only count PRIME_VISIBLE properties (value, borders, fill)
                    prime_visible_mismatches = filter_to_prime_visible(filtered_mismatches)

                attempt_meta: Dict[str, Any] = {
                    "retry": retry,
                    "prompt_text": self._format_messages(messages),
                    "response_text": response_text,
                    "parsed_operations": len(parsed.operations),
                    "valid_operations": len(valid_ops),
                    "invalid_operations": len(invalid_ops),
                    "skipped_operations": len(skipped_ops),
                    "rejected_operations": len(rejected),
                    "skip_declarations": len(parsed.skip_declarations),
                    "ignore_declarations": len(parsed.ignore_declarations.declarations),
                    "human_enough": parsed.human_enough,
                    "rationale": parsed.rationale,
                    "tokens": response.usage.model_dump() if response.usage else {},
                    "prompt_tokens": usage_snapshot["prompt"],
                    "completion_tokens": usage_snapshot["completion"],
                    "total_tokens": usage_snapshot["total"],
                    "elapsed_seconds": attempt_elapsed,
                    "validation_errors": list(validation_errors),
                }
                if comparison is not None:
                    attempt_meta["mismatched_cell_count"] = len(mismatched_cells)
                    attempt_meta["skipped_mismatch_count"] = skipped_count
                    attempt_meta["filtered_mismatch_count"] = len(filtered_mismatches)
                retry_history.append(attempt_meta)
                call_meta = {k: v for k, v in attempt_meta.items() if k not in {"prompt_text", "response_text"}}
                self._record_llm_call(
                    call_type="refiner",
                    iteration=iteration,
                    retry=retry,
                    label=f"iteration_{iteration}_retry_{retry}",
                    messages=messages,
                    response_text=response_text,
                    usage=usage_snapshot,
                    elapsed=attempt_elapsed,
                    metadata=call_meta,
                )

                if validation_errors:
                    retry_feedback_text = build_retry_feedback_message(
                        retry=retry,
                        max_retries=self.config.max_retries,
                        validation_errors=validation_errors,
                        iteration=iteration,
                    )
                    retry_response_text = response_text
                    continue

                if comparison is None:
                    retry_feedback_text = build_retry_feedback_message(
                        retry=retry,
                        max_retries=self.config.max_retries,
                        validation_errors=["Comparison failed unexpectedly"],
                        iteration=iteration,
                    )
                    retry_response_text = response_text
                    continue

                # Score is based on PRIME_VISIBLE properties only (value, borders, fill)
                # Other formatting differences don't fail validation
                score = len(prime_visible_mismatches)
                if parsed and comparison:
                    if best_attempt_score is None or score < best_attempt_score:
                        best_attempt = (
                            valid_ops.copy(),
                            comparison,
                            parsed,
                            response_text,
                            completion_moves.copy(),
                            comparison_report,
                            incorrect_operations.copy(),
                        )
                        best_attempt_score = score

                if score > 0:
                    mismatch_report = comparison_report or "Mismatched cells unavailable."
                    retry_feedback_text = build_retry_mismatch_message(
                        retry=retry,
                        max_retries=self.config.max_retries,
                        mismatch_report=mismatch_report,
                        repair_preview=completion_moves,
                        mismatch_operations=incorrect_operations,
                        iteration=iteration,
                    )
                    retry_response_text = response_text
                    continue

                inner_success = True
                candidate_ops = valid_ops.copy()
                candidate_parsed = parsed
                candidate_comparison = comparison
                candidate_report = comparison_report
                break
            else:
                # Exhausted retries without obtaining a valid candidate
                validation_errors = list(validation_errors)

            if not inner_success and best_attempt:
                valid_ops, comparison, parsed, response_text, completion_moves, comparison_report, incorrect_operations = best_attempt
                mismatched_cells = list_mismatched_cells(comparison)
                validation_errors = []

            if service_down_reason:
                break

            # Re-compress valid_ops for diff summary comparison (if compression enabled)
            valid_ops_for_diff = valid_ops
            if self.config.compress_inputs:
                valid_ops_for_diff, _ = compress_operations(
                    valid_ops,
                    cell_threshold=self.config.input_compression_threshold,
                    string_threshold=self.config.input_string_truncate_threshold,
                )
            diff_summary = summarize_sequence_diff(valid_ops_for_diff, compressed_ops, self.config.feedback_diff_limit)
            comparator_summary = summarize_comparison(comparison, self.config.feedback_cell_limit)
            retry_hint = build_retry_hint(None if validation_errors else comparison)
            extra_summary = summarize_extra_properties(comparison, limit=3)
            iteration_meta: Dict[str, Any] = {
                "parsed_operations": len(parsed.operations) if parsed else 0,
                "valid_operations": len(valid_ops),
                "invalid_operations": len(invalid_ops),
                "skipped_operations": len(skipped_ops),
                "rejected_operations": len(rejected),
                "human_enough": parsed.human_enough if parsed else False,
                "rationale": parsed.rationale if parsed else "",
                "tokens": response.usage.model_dump() if response.usage else {},
                "validation_errors": validation_errors,
                "retry_attempts": len(retry_history),
                "retry_history": retry_history,
                "completion_moves": completion_moves,
                "ops_to_reach_target": getattr(comparison, "ops_diff", None) if comparison else None,
                "inner_success": inner_success,
                "token_usage": iteration_usage,
                "elapsed_seconds": iteration_elapsed,
            }
            if comparison:
                iteration_meta.update(
                    {
                        "mismatched_cells": mismatched_cells,
                        "mismatched_cell_count": len(mismatched_cells),
                    }
                )
                self._log_progress(
                    f"Iteration {iteration} completed with {len(mismatched_cells)} mismatched cells"
                )
            else:
                self._log_progress(f"Iteration {iteration} completed without a valid comparison")

            iteration_summary = self._build_attempt_summary(
                iteration=iteration,
                parsed=parsed,
                validation_errors=validation_errors,
                comparator_summary=comparator_summary,
                diff_summary=diff_summary,
                completion_moves=completion_moves,
            )
            if iteration_summary:
                iteration_meta["attempt_summary"] = iteration_summary
                self._update_history(history_summaries, iteration_summary)

            judge_approved = False
            if not inner_success:
                feedback_message = build_feedback_message(
                    iteration=iteration,
                    max_iterations=self.config.max_iterations,
                    validation_errors=validation_errors,
                    comparator_summary=comparator_summary,
                    diff_summary=diff_summary,
                    retry_hint=retry_hint,
                    completion_moves=completion_moves,
                    ops_to_reach_target=(getattr(comparison, "ops_diff", None) if comparison else None),
                    extra_summary=extra_summary,
                )
                iteration_meta["iteration_feedback"] = feedback_message
            else:
                judge_label = f"Iteration {iteration} candidate"
                # Re-compress candidate ops for judge (token efficiency)
                candidate_ops_for_judge = candidate_ops or []
                if self.config.compress_inputs:
                    candidate_ops_for_judge, _ = compress_operations(
                        candidate_ops_for_judge,
                        cell_threshold=self.config.input_compression_threshold,
                        string_threshold=self.config.input_string_truncate_threshold,
                    )
                iteration_diff_summary = summarize_sequence_diff(
                    candidate_ops_for_judge,
                    compressed_ops,
                    limit=None,
                )
                verdict, judge_usage, judge_elapsed = self._run_judge(
                    candidate_ops_for_judge,
                    judge_label,
                    f"iteration_{iteration}",
                    iteration,
                    sheet_image,
                    iteration_diff_summary,
                )
                judge_history.append(self._judge_record(verdict, iteration_label=f"iteration_{iteration}"))
                for key in iteration_usage:
                    iteration_usage[key] += judge_usage[key]
                    overall_usage[key] += judge_usage[key]
                iteration_elapsed += judge_elapsed
                overall_elapsed += judge_elapsed
                self._log_progress(
                    "Judge review tokens — prompt: {prompt}, completion: {completion}, total: {total}, elapsed: {elapsed:.2f}s".format(
                        prompt=judge_usage["prompt"],
                        completion=judge_usage["completion"],
                        total=judge_usage["total"],
                        elapsed=judge_elapsed,
                    )
                )
                iteration_meta["judge_verdict"] = self._judge_record(
                    verdict,
                    iteration_label=f"iteration_{iteration}",
                    include_raw=True,
                )
                iteration_meta["judge_verdict"].update(
                    {
                        "token_usage": judge_usage,
                        "elapsed_seconds": judge_elapsed,
                        "diff_summary": iteration_diff_summary,
                    }
                )
                last_feedback_text = wrap_judge_feedback(verdict.raw_text)
                last_response_text = response_text
                self._update_history(
                    history_summaries,
                    f"Judge iter {iteration}: {'\u2705' if verdict.is_human else '\u26a0\ufe0f'} {verdict.rationale[:80]}",
                )
                if verdict.is_human and candidate_ops and candidate_comparison and candidate_parsed:
                    judge_approved = True
                    accepted_candidate = candidate_ops.copy()
                    best_candidate = (candidate_ops.copy(), candidate_comparison, candidate_parsed)
                    best_score = 0
                    best_comparator_report = candidate_report
                    best_mismatched_cells = []
                    judge_rejection_reason = None
                else:
                    judge_rejection_reason = (
                        f"Judge rejected iteration {iteration} candidate despite workbook parity"
                    )

            self._log_progress(
                "Iteration {iter_idx} totals — prompt: {prompt}, completion: {completion}, total: {total}, elapsed: {elapsed:.2f}s".format(
                    iter_idx=iteration,
                    prompt=iteration_usage["prompt"],
                    completion=iteration_usage["completion"],
                    total=iteration_usage["total"],
                    elapsed=iteration_elapsed,
                )
            )
            save_iteration_artifacts(
                run_dir,
                iteration,
                prompt_text=retry_history[-1]["prompt_text"] if retry_history else "",
                response_text=response_text,
                metadata=iteration_meta,
                comparator_report=comparison_report,
            )

            if comparison and parsed:
                score = len(mismatched_cells)
                if best_score is None or score < best_score:
                    best_candidate = (valid_ops.copy(), comparison, parsed)
                    best_score = score
                    best_comparator_report = comparison_report
                    best_mismatched_cells = mismatched_cells.copy()

            if judge_approved:
                break

            if service_down_reason:
                self._log_progress("Stopping early due to LLM outage")
                break

        summary = {
            "success": bool(accepted_candidate),
            "iterations": iterations_run,
            "best_score": best_score,
            "sheet": self.config.sheet_name,
            "max_dimension": self.config.max_dimension,
            "mismatched_cells": best_mismatched_cells,
            "overall_token_usage": overall_usage,
            "overall_elapsed_seconds": overall_elapsed,
            "baseline_accepted": False,
            "judge_history": judge_history,
        }
        if accepted_candidate:
            summary["accepted_sequence_source"] = "iteration"
        if service_down_reason:
            summary["failure_reason"] = service_down_reason
        elif judge_rejection_reason:
            summary["failure_reason"] = judge_rejection_reason

        self._log_progress(
            "Overall tokens — prompt: {prompt}, completion: {completion}, total: {total}, elapsed: {elapsed:.2f}s".format(
                prompt=overall_usage["prompt"],
                completion=overall_usage["completion"],
                total=overall_usage["total"],
                elapsed=overall_elapsed,
            )
        )

        best_ops_text = None
        best_ops_list: List[str] = []
        if accepted_candidate:
            best_ops_list = accepted_candidate
            best_ops_text = "\n".join(best_ops_list)
        elif best_candidate:
            best_ops_list = best_candidate[0]
            best_ops_text = "\n".join(best_ops_list)
        write_final_report(run_dir, summary, best_comparator_report, best_ops_text)
        save_llm_call_log(run_dir, self.llm_call_log)

        if summary["success"]:
            message = "Refinement succeeded"
        elif service_down_reason:
            message = service_down_reason
        elif judge_rejection_reason:
            message = judge_rejection_reason
        else:
            message = "Refinement failed"
        return RefinementOutcome(summary["success"], summary["iterations"], best_ops_list, str(run_dir), message)

    @staticmethod
    def _format_messages(messages: List[Message]) -> str:
        lines = []
        for msg in messages:
            label = msg.role.value.upper()
            lines.append(f"[{label}] {msg.content}")
        return "\n\n".join(lines)

    def _build_messages(
        self,
        system_prompt: str,
        instruction: str,
        sheet_image: Optional[Path],
        last_response_text: Optional[str],
        last_feedback_text: Optional[str],
        last_retry_response_text: Optional[str],
        last_retry_feedback_text: Optional[str],
        history_summary: Optional[str],
    ) -> List[Message]:
        messages: List[Message] = []
        messages.append(Message(role=Role.System, content=system_prompt))
        if sheet_image:
            messages.append(
                Message(
                    role=Role.User,
                    content=instruction,
                    image=str(sheet_image),
                )
            )
        else:
            messages.append(
                Message(
                    role=Role.User,
                    content=instruction,
                )
            )
        if last_retry_response_text:
            messages.append(Message(role=Role.Assistant, content=last_retry_response_text))
        elif last_response_text:
            messages.append(Message(role=Role.Assistant, content=last_response_text))
        if history_summary:
            messages.append(Message(role=Role.User, content=history_summary))
        if last_feedback_text:
            messages.append(Message(role=Role.User, content=last_feedback_text))
        if last_retry_feedback_text:
            messages.append(Message(role=Role.User, content=last_retry_feedback_text))
        return messages

    def _build_request_overrides(self, retry_index: int) -> Dict[str, Any]:
        """Adjust temperature/effort per retry, matching structured retry logic."""
        attempt_number = max(retry_index - 1, 0)
        temperature = 0.0 if attempt_number == 0 else 0.1 + attempt_number * 0.05

        overrides: Dict[str, Any] = {"temperature": temperature}

        if getattr(self.llm, "supports_reasoning", False):
            overrides["reasoning_effort"] = self.config.reasoning_effort or "low"
            overrides["temperature"] = 1.0

        return overrides

    def _complete_with_service_check(
        self,
        adapter: RefinementLLMAdapter,
        messages: List[Message],
        request_overrides: Optional[Dict[str, Any]],
    ) -> LLMResponse:
        attempts = self.config.api_down_retry_attempts
        delay = max(self.config.api_down_retry_delay, 0.0)
        for attempt in range(attempts + 1):
            try:
                return adapter.complete(messages, request_overrides=request_overrides)
            except LLMServiceUnavailableError as exc:
                if attempt >= attempts:
                    raise
                logger.warning("LLM outage detected, retrying (%s/%s)", attempt + 1, attempts + 1)
                if delay:
                    time.sleep(delay)
        raise RuntimeError("Unreachable path in _complete_with_service_check")

    def _log_progress(self, message: str) -> None:
        if not self.config.log_progress:
            return
        logger.info(message)
        print(message)

    def _derive_repair_operations(
        self,
        predicted_state: Optional[Dict[str, Any]],
        target_state: Dict[str, Any],
    ) -> List[str]:
        """Derive symbolic operations needed to move from predicted to target state.

        This is a simplified version that uses the comparator's differences
        to suggest repair operations, without relying on the old excel_converter
        parse_excel/merge_operations pipeline.
        """
        limit = self.config.repair_operation_limit
        if predicted_state is None or target_state is None:
            return []
        if limit is not None and limit <= 0:
            return []

        comparison = self.comparator.compare(predicted_state, target_state)
        moves = build_completion_moves(comparison.differences, limit=limit)
        return moves

    def _format_history_summary(self, summaries: List[str]) -> Optional[str]:
        if not summaries:
            return None
        header = "Recent attempt recap:"
        body = "\n".join(f"- {entry}" for entry in summaries)
        return f"{header}\n{body}"

    def _update_history(self, history: List[str], entry: str) -> None:
        limit = self.config.history_summary_limit
        if limit is None:
            history.append(entry)
            return
        if limit <= 0:
            history.clear()
            return
        history.append(entry)
        excess = len(history) - limit
        if excess > 0:
            del history[:excess]

    @staticmethod
    def _build_attempt_summary(
        *,
        iteration: int,
        parsed: Optional[ParsedLLMOutput],
        validation_errors: Iterable[str],
        comparator_summary: str,
        diff_summary: str,
        completion_moves: List[str],
    ) -> str:
        validation_list = list(validation_errors)
        human_status = "yes" if parsed and parsed.human_enough else "no/unknown"
        if validation_list:
            limited = "; ".join(validation_list[:2])
            if len(validation_list) > 2:
                limited += "; ..."
            validation = f"issues: {limited}"
        else:
            validation = "passed"
        if completion_moves:
            preview_ops = ", ".join(completion_moves[:2])
            if len(completion_moves) > 2:
                preview_ops += ", ..."
        else:
            preview_ops = "n/a"
        rationale = (parsed.rationale.strip().splitlines()[0] if parsed and parsed.rationale else "")
        rationale_part = f" rationale={rationale}" if rationale else ""
        return (
            f"[Attempt {iteration}] human={human_status}; validation={validation}; "
            f"comparison={comparator_summary}; delta={diff_summary}; repairs={preview_ops}{rationale_part}"
        )

    def _run_judge(
        self,
        operations: List[str],
        sequence_title: str,
        iteration_label: str,
        iteration_index: Optional[int],
        sheet_image: Optional[Path],
        diff_summary: str,
    ) -> Tuple[JudgeVerdict, Dict[str, int], float]:
        operations_block = format_operations_block(operations, limit=None)
        prompt = build_judge_prompt(
            sheet_name=self.config.sheet_name,
            sequence_title=sequence_title,
            operations_block=operations_block,
            diff_summary=diff_summary,
            judge_keyword=self.config.judge_keyword,
        )
        user_msg_kwargs: Dict[str, Any] = {"role": Role.User, "content": prompt}
        if sheet_image:
            user_msg_kwargs["image"] = str(sheet_image)
        messages = [
            Message(role=Role.System, content=build_judge_system_prompt()),
            Message(**user_msg_kwargs),
        ]
        overrides: Dict[str, Any] = {}
        if getattr(self.judge, "supports_reasoning", False):
            overrides["reasoning_effort"] = self.config.judge_reasoning_effort or "high"
            overrides["temperature"] = 1.0
        else:
            overrides["temperature"] = self.config.judge_temperature
        start = time.perf_counter()
        response = self._complete_with_service_check(self.judge, messages, overrides)
        elapsed = time.perf_counter() - start
        text = response.text or ""
        usage = self._extract_token_usage(response.usage)

        # Warn if judge response is empty (possible token limit exhaustion)
        if not text.strip():
            logger.warning(
                "Judge returned empty response (completion tokens: %s). "
                "This may indicate judge_max_completion_tokens (%s) is too low for reasoning models.",
                usage.get("completion", 0),
                self.config.judge_max_completion_tokens,
            )

        verdict = parse_judge_response(text, self.config.judge_keyword)
        self._record_llm_call(
            call_type="judge",
            iteration=iteration_index,
            retry=None,
            label=iteration_label,
            messages=messages,
            response_text=text,
            usage=usage,
            elapsed=elapsed,
            metadata={
                "sequence_title": sequence_title,
                "is_human": verdict.is_human,
                "rationale": verdict.rationale,
                "diff_summary": diff_summary,
            },
        )
        return verdict, usage, elapsed

    @staticmethod
    def _judge_record(
        verdict: JudgeVerdict,
        *,
        iteration_label: str,
        include_raw: bool = False,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "label": iteration_label,
            "is_human": verdict.is_human,
            "rationale": verdict.rationale,
        }
        if include_raw:
            record["raw_text"] = verdict.raw_text
        return record

    @staticmethod
    def _format_cell_preview(cells: Iterable[str], limit: int) -> str:
        snapshot = list(cells)
        if not snapshot:
            return "(all cells aligned)"
        preview = snapshot[:limit]
        suffix = "" if len(snapshot) <= limit else " ..."
        return ", ".join(preview) + suffix

    @staticmethod
    def _extract_token_usage(usage: Any) -> Dict[str, int]:
        snapshot = {"prompt": 0, "completion": 0, "total": 0}
        if not usage:
            return snapshot
        snapshot["prompt"] = getattr(usage, "prompt_tokens", 0) or 0
        snapshot["completion"] = getattr(usage, "completion_tokens", 0) or 0
        snapshot["total"] = getattr(usage, "total_tokens", 0) or (
            snapshot["prompt"] + snapshot["completion"]
        )
        return snapshot

    def _record_llm_call(
        self,
        *,
        call_type: str,
        iteration: Optional[int],
        retry: Optional[int],
        label: str,
        messages: List[Message],
        response_text: str,
        usage: Dict[str, int],
        elapsed: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        serialized_messages = []
        for msg in messages:
            entry: Dict[str, Any] = {"role": msg.role.value, "content": msg.content}
            image = getattr(msg, "image", None)
            if image:
                entry["image"] = image
            serialized_messages.append(entry)
        record: Dict[str, Any] = {
            "type": call_type,
            "iteration": iteration,
            "retry": retry,
            "label": label,
            "messages": serialized_messages,
            "prompt_text": self._format_messages(messages),
            "response_text": response_text,
            "usage": usage,
            "elapsed_seconds": elapsed,
        }
        if metadata:
            record["metadata"] = metadata
        self.llm_call_log.append(record)
        self.call_log_index += 1
        if self.call_log_dir:
            file_stem = f"{self.call_log_index:03d}"
            json_path = self.call_log_dir / f"{file_stem}.json"
            json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

            safe_label = self._sanitize_label(label)
            call_dir = self.call_log_dir / f"{file_stem}_{safe_label}"
            call_dir.mkdir(parents=True, exist_ok=True)

            prompt_text = record.get("prompt_text", "")
            (call_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
            (call_dir / "response.txt").write_text(response_text or "", encoding="utf-8")

            feedback_sections: List[str] = []
            first_user_index: Optional[int] = None
            for idx, msg in enumerate(messages):
                if msg.role == Role.User:
                    first_user_index = idx
                    break
            for idx, msg in enumerate(messages):
                if msg.role != Role.User:
                    continue
                if first_user_index is not None and idx == first_user_index:
                    continue
                feedback_sections.append(msg.content.strip())
            if metadata:
                metadata_dump = json.dumps(metadata, indent=2)
                feedback_sections.append(f"[metadata]\n{metadata_dump}")
            feedback_payload = "\n\n-----\n\n".join(section for section in feedback_sections if section)
            if not feedback_payload:
                feedback_payload = "No feedback messages captured."
            (call_dir / "feedbacks.txt").write_text(feedback_payload, encoding="utf-8")

    @staticmethod
    def _sanitize_label(label: str) -> str:
        base = label.strip() or "call"
        cleaned = [ch if ch.isalnum() or ch in {"_", "-"} else "-" for ch in base]
        text = "".join(cleaned).strip("-")
        return text or "call"
