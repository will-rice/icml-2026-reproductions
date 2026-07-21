from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def create_run_directory(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "iterations").mkdir(parents=True, exist_ok=True)
    return base_dir


def save_iteration_artifacts(
    run_dir: Path,
    iteration: int,
    prompt_text: str,
    response_text: str,
    metadata: Dict[str, Any],
    comparator_report: Optional[str],
) -> None:
    iter_dir = run_dir / "iterations" / f"{iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    (iter_dir / "response.txt").write_text(response_text, encoding="utf-8")
    if comparator_report:
        (iter_dir / "comparison.txt").write_text(comparator_report, encoding="utf-8")
    (iter_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_final_report(
    run_dir: Path,
    summary: Dict[str, Any],
    comparator_report: Optional[str],
    best_operations: Optional[str],
) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if best_operations:
        (run_dir / "best_operations.txt").write_text(best_operations, encoding="utf-8")
    report_lines = ["# Sequence Refinement Report", ""]
    report_lines.append(f"Status: {'SUCCESS' if summary.get('success') else 'FAILED'}")
    report_lines.append(f"Iterations: {summary.get('iterations')}")
    report_lines.append("")
    if comparator_report:
        report_lines.append("## Last comparator report")
        report_lines.append("")
        report_lines.append(comparator_report)
    (run_dir / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def save_llm_call_log(run_dir: Path, calls: List[Dict[str, Any]]) -> None:
    if not calls:
        return
    calls_dir = run_dir / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    for idx, entry in enumerate(calls, start=1):
        path = calls_dir / f"{idx:03d}.json"
        path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
