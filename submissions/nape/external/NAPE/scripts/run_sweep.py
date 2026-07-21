#!/usr/bin/env python
"""
Sweep Runner

Loops over multiple experiment configs, expanding sweeps, and delegates
to run_batch_evaluation for each variant.

Usage:
    # Explicit config list (ad-hoc runs)
    python scripts/run_sweep.py configs/evaluation/stride1.yaml configs/evaluation/stride3.yaml

    # Single config with sweep section
    python scripts/run_sweep.py configs/evaluation/sweep_context.yaml

    # Mix: multiple configs, some with sweeps
    python scripts/run_sweep.py configs/evaluation/*.yaml --output-dir outputs/full_sweep
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import List

# Add project root and src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from next_action_pred_eval.evaluation.experiment_config import (
    load_experiment_config,
    expand_sweep,
    ExperimentConfig,
)
from next_action_pred_eval.evaluation.output_layout import TrajectoryResult

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run experiment sweep")
    parser.add_argument("configs", nargs="+", type=Path, help="Config YAML files")
    parser.add_argument("--output-dir", type=Path, help="Override base output directory")
    parser.add_argument("--no-resume", action="store_true", help="Force re-run all")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Expand all configs into variants
    all_variants: List[ExperimentConfig] = []
    for config_path in args.configs:
        if not config_path.exists():
            logger.warning("Config file not found, skipping: %s", config_path)
            continue
        base_cfg = load_experiment_config(config_path)
        variants = expand_sweep(base_cfg)
        if args.output_dir:
            for v in variants:
                v.output_dir = str(args.output_dir / v.variant_name)
        else:
            # Auto-name: put each variant in its own subdir under base output_dir
            base_output = base_cfg.output_dir
            for v in variants:
                v.output_dir = str(Path(base_output) / v.variant_name)
        all_variants.extend(variants)

    logger.info(
        "Sweep: %d total config variants from %d config files",
        len(all_variants), len(args.configs),
    )

    if not all_variants:
        logger.error("No valid configs found.")
        return 1

    # Set up sweep-level log
    sweep_output = args.output_dir or Path(all_variants[0].output_dir).parent
    sweep_output = Path(sweep_output)
    sweep_output.mkdir(parents=True, exist_ok=True)
    sweep_log_path = sweep_output / "sweep.log"

    file_handler = logging.FileHandler(sweep_log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Import here to avoid circular import at module level
    from scripts.run_batch_evaluation import run_experiment_config

    sweep_start = time.time()
    all_results: List[TrajectoryResult] = []

    for i, variant in enumerate(all_variants):
        logger.info("=" * 60)
        logger.info("Variant %d/%d: %s", i + 1, len(all_variants), variant.variant_name)
        logger.info("=" * 60)

        variant_results = run_experiment_config(variant, resume=not args.no_resume)
        all_results.extend(variant_results)

        successful = sum(1 for r in variant_results if r.status == "success")
        logger.info(
            "Variant %s complete: %d/%d successful",
            variant.variant_name, successful, len(variant_results),
        )

    sweep_time = time.time() - sweep_start

    # Write sweep_summary.csv
    if all_results:
        from next_action_pred_eval.evaluation.output_layout import CSV_COLUMNS
        sweep_csv_path = sweep_output / "sweep_summary.csv"
        with open(sweep_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r.to_csv_row())
        logger.info("Wrote sweep summary: %s (%d rows)", sweep_csv_path, len(all_results))

    logger.info("Sweep complete: %d variants, %.1fs", len(all_variants), sweep_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
