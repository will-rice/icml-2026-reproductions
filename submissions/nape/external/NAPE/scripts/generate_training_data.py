"""
Full Pipeline Script for Training Data Generation.

Runs a complete 6-step pipeline for each workbook/sheet target:
1. Screenshot Generation  - Captures sheet images (via xlwings)
2. Region Analysis        - Uses LLM to identify structured regions
3. Heuristics Sampling    - Samples sequencing configurations
4. Sequence Generation    - Creates operation sequences using sampled heuristics
5. Sequence Refinement    - Refines sequences using LLM feedback
6. GIF Generation         - Creates GIF from refined sequence (if successful)

Each step is completed for ALL files before moving to the next step.
Outputs are organized in: OUTPUT_DIR/{file_prefix}_{sheet_name}/

Dependencies:
  - next_action_pred_eval.utils.llm.base.LLMAdapter  (LLM calls)
  - next_action_pred_eval.utils.workbook.excel_parser.ExcelParser (xlsx -> operations)
  - next_action_pred_eval.generation.regions           (region analysis)
  - next_action_pred_eval.generation.refinement        (refinement)
  - next_action_pred_eval.generation.sequencing        (sequencing engine)
  - next_action_pred_eval.generation.sequence          (prompts / validation)

Usage:
    python generate_training_data.py --targets targets.json --output outputs/run_1
    python generate_training_data.py --targets targets.json --output outputs/run_1 --model gpt-4o
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Ensure the package is importable when running as a script
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------------------------------------------------------------------------
# Internal imports (next_action_pred_eval only, no banned dependencies)
# ---------------------------------------------------------------------------
from next_action_pred_eval.utils.workbook.excel_parser import ExcelParser
from next_action_pred_eval.core.operations import MergeCells
from next_action_pred_eval.core.symbolic import operations_to_symbolic, symbolic_to_operations
from next_action_pred_eval.generation.regions import analyze_sheet_regions, StructuredRegionOutput
from next_action_pred_eval.generation.sequencing import SequencingEngine
from next_action_pred_eval.generation.sequencing.config_sampler import sample_config
from next_action_pred_eval.generation.refinement import (
    RefinementConfig,
    SequenceRefinementPipeline,
)
from next_action_pred_eval.utils.llm.base import LLMAdapter

# Configure logging - will be fully set up in run_pipeline()
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================

DEFAULT_OUTPUT_DIR = Path("outputs/training_data")
DEFAULT_MAX_DIMENSION = 200
DEFAULT_USE_CACHE = True
DEFAULT_CACHE_DIR = Path("caches")

# Region analysis
DEFAULT_REGION_ANALYSIS_MAX_RETRIES = 3

# Refinement
DEFAULT_REFINEMENT_MAX_ITERATIONS = 3
DEFAULT_REFINEMENT_MAX_RETRIES = 3
DEFAULT_REFINEMENT_TEMPERATURE = 0.15
DEFAULT_REFINEMENT_REASONING_EFFORT = "low"
DEFAULT_REFINEMENT_MAX_COMPLETION_TOKENS = None
DEFAULT_JUDGE_REASONING_EFFORT = "high"
DEFAULT_JUDGE_MAX_COMPLETION_TOKENS = 20000

# Config sampler
DEFAULT_CONFIG_SAMPLER_SEED = None

# GIF
DEFAULT_GIF_DURATION_MS = 300
DEFAULT_GIF_LOOP = 1
DEFAULT_MAX_OP_LEN_FOR_GIF: Optional[int] = 1000


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TargetInfo:
    """Information about a processing target."""
    workbook_path: Path
    sheet_name: str
    output_name: str  # "{file_prefix}_{sheet_name}"
    output_dir: Path

    # Paths populated during pipeline
    sheet_image_path: Optional[Path] = None
    region_output_path: Optional[Path] = None
    sampled_config: Optional[Dict[str, Any]] = None
    sampled_config_path: Optional[Path] = None
    sequence_output_path: Optional[Path] = None
    framework_output_path: Optional[Path] = None
    refinement_output_dir: Optional[Path] = None
    refinement_success: Optional[bool] = None
    gif_path: Optional[Path] = None

    # Status tracking
    skipped: bool = False
    skip_reason: Optional[str] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Result of the full pipeline run."""
    total_targets: int
    successful: int
    failed: int
    skipped: int
    targets: List[TargetInfo]
    errors: List[Dict[str, Any]]
    skipped_list: List[Dict[str, str]]
    start_time: datetime
    end_time: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def force_garbage_collection(reason: str = "") -> None:
    """Force garbage collection and log memory cleanup."""
    collected = gc.collect()
    if collected > 0:
        logger.debug(f"GC collected {collected} objects{' (' + reason + ')' if reason else ''}")


def sanitize_name(name: str) -> str:
    """Sanitize a name for use in file/folder names."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def get_output_name(workbook_path: Path, sheet_name: str) -> str:
    """Generate output name: {file_prefix}_{sheet_name}."""
    file_prefix = workbook_path.stem.split("-")[0]
    return f"{file_prefix}_{sanitize_name(sheet_name)}"


def is_file_valid(file_path: Path, min_size: int = 0) -> bool:
    """Check if a file exists and is valid (not empty/corrupted)."""
    if not file_path.exists():
        return False
    try:
        size = file_path.stat().st_size
        if size <= min_size:
            return False
        # For JSON files, try to parse them
        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                json.load(f)
        return True
    except (json.JSONDecodeError, OSError, IOError):
        return False


def is_image_valid(image_path: Path) -> bool:
    """Check if an image file exists and is valid."""
    if not image_path.exists():
        return False
    try:
        size = image_path.stat().st_size
        if size < 100:
            return False
        with open(image_path, "rb") as f:
            header = f.read(8)
            return header[:4] == b"\x89PNG"
    except (OSError, IOError):
        return False


def is_gif_valid(gif_path: Path) -> bool:
    """Check if a GIF file exists and is valid."""
    if not gif_path.exists():
        return False
    try:
        size = gif_path.stat().st_size
        if size < 100:
            return False
        with open(gif_path, "rb") as f:
            header = f.read(6)
            return header in (b"GIF87a", b"GIF89a")
    except (OSError, IOError):
        return False


def cleanup_incomplete_file(file_path: Path) -> None:
    """Remove an incomplete/corrupted file if it exists."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up incomplete file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup {file_path}: {e}")


def restore_target_state(target: TargetInfo) -> None:
    """Restore the state of a target from previous runs."""
    # Check for valid screenshot
    screenshot_path = target.output_dir / "sheet_image.png"
    if is_image_valid(screenshot_path):
        target.sheet_image_path = screenshot_path

    # Check for valid region analysis
    region_path = target.output_dir / "regions" / "output.json"
    if is_file_valid(region_path, min_size=10):
        target.region_output_path = region_path

    # Check for valid sampled config
    config_path = target.output_dir / "sampled_configs" / "sampled_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            if isinstance(config_data, dict) and len(config_data) > 0:
                target.sampled_config = config_data
                target.sampled_config_path = config_path
        except (json.JSONDecodeError, IOError):
            pass

    # Check for valid sequence
    sequence_path = target.output_dir / "sequences" / "framework_output.txt"
    if is_file_valid(sequence_path, min_size=10):
        target.sequence_output_path = target.output_dir / "sequences"
        target.framework_output_path = sequence_path

    # Check for valid refinement
    refinement_dir = target.output_dir / "refinement"
    if refinement_dir.exists():
        summary_file = refinement_dir / "summary.json"
        if is_file_valid(summary_file, min_size=10):
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                target.refinement_output_dir = refinement_dir
                target.refinement_success = summary.get("success", False)
            except (json.JSONDecodeError, IOError):
                pass

    # Check for valid GIF
    if target.refinement_output_dir:
        gif_path = target.refinement_output_dir / "operation_sequence.gif"
        if is_gif_valid(gif_path):
            target.gif_path = gif_path

    if not target.gif_path and target.sequence_output_path:
        gif_path = target.sequence_output_path / "operation_sequence.gif"
        if is_gif_valid(gif_path):
            target.gif_path = gif_path


def is_target_fully_complete(target: TargetInfo) -> bool:
    """Check if a target has completed all pipeline steps successfully."""
    if target.skipped:
        return True
    if target.errors:
        return False
    if not target.sheet_image_path:
        return False
    if not target.region_output_path:
        return False
    if not target.sampled_config:
        return False
    if not target.framework_output_path:
        return False
    if target.refinement_output_dir is None:
        return False
    if not target.gif_path:
        return False
    return True


def find_bounding_boxes(
    operations: List,
    max_sheet_limit: int = 200,
) -> Dict[str, str]:
    """Find bounding boxes from a list of operations.

    Groups operations by contiguous rectangular regions and returns
    a dict mapping region labels to range strings.
    """
    if not operations:
        return {}

    from next_action_pred_eval.utils.cell_utils import get_range_string

    # Collect all coordinates
    cells = []
    for op in operations:
        try:
            coords = op.cell_range.get_coordinates()
            sr, sc, er, ec = coords
            if er <= max_sheet_limit and ec <= max_sheet_limit:
                cells.append((sr, sc, er, ec))
        except Exception:
            continue

    if not cells:
        return {}

    # Simple bounding-box: one region covering everything
    min_r = min(c[0] for c in cells)
    min_c = min(c[1] for c in cells)
    max_r = min(max(c[2] for c in cells), max_sheet_limit)
    max_c = min(max(c[3] for c in cells), max_sheet_limit)

    range_str = get_range_string(min_r, min_c, max_r, max_c)
    return {"region_1": range_str}


def create_gif_from_images(
    image_paths: List[str],
    output_path: str,
    duration: int = 300,
    loop: int = 1,
) -> None:
    """Create a GIF from a list of image file paths.

    Uses Pillow (PIL) to compose frames.
    """
    from PIL import Image

    if not image_paths:
        return

    frames = []
    for path in image_paths:
        img = Image.open(path)
        frames.append(img.copy())
        img.close()

    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop,
        )

    for frame in frames:
        frame.close()


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def should_skip_target(target: TargetInfo, step_name: str) -> Tuple[bool, Optional[str]]:
    """Check if a target should be skipped."""
    if target.skipped:
        return True, "already marked as skipped"
    if target.errors:
        return True, f"previous errors: {target.errors[0][:50]}..."
    return False, None


def step1_generate_screenshots(
    targets: List[TargetInfo],
    max_dimension: int,
) -> None:
    """Step 1: Generate screenshots for all sheets using xlwings."""
    print("\n" + "=" * 80)
    print("STEP 1: GENERATING SCREENSHOTS")
    print("=" * 80)

    # xlwings is optional; if not available screenshots must be pre-generated
    try:
        import xlwings as xw
    except ImportError:
        xw = None
        print("  WARNING: xlwings not available. Screenshots must be pre-generated.")

    for target in targets:
        skip, reason = should_skip_target(target, "screenshot")
        if skip:
            print(f"  [skip] {target.output_name} ({reason})")
            continue

        existing_screenshot = target.output_dir / "sheet_image.png"
        if is_image_valid(existing_screenshot):
            target.sheet_image_path = existing_screenshot
            print(f"  [ok] Already processed: {target.output_name}")
            continue

        cleanup_incomplete_file(existing_screenshot)
        print(f"  Processing: {target.output_name}")

        if xw is None:
            target.errors.append("Screenshot generation failed: xlwings not installed")
            print(f"    [error] xlwings not installed")
            continue

        try:
            app = xw.App(visible=False, add_book=False)
            try:
                wb = app.books.open(str(target.workbook_path))
                sht = wb.sheets[target.sheet_name]

                # Determine used range
                used = sht.used_range
                if used is None:
                    raise ValueError("Sheet has no used range")

                # Capture image
                target.output_dir.mkdir(parents=True, exist_ok=True)
                output_path = target.output_dir / "sheet_image.png"

                # Use range screenshot via xlwings
                last_row = min(used.last_cell.row, max_dimension)
                last_col = min(used.last_cell.column, max_dimension)
                capture_range = sht.range((1, 1), (last_row, last_col))

                capture_range.to_png(str(output_path))

                target.sheet_image_path = output_path
                print(f"    [ok] Generated: {output_path.name}")

                wb.close()
            finally:
                app.quit()

        except Exception as e:
            error_msg = f"Screenshot generation failed: {str(e)}"
            target.errors.append(error_msg)
            print(f"    [error] {error_msg}")
            logger.error(f"Error generating screenshot for {target.output_name}: {e}", exc_info=True)

    force_garbage_collection("after step 1 screenshots")
    print(f"\nStep 1 complete: {sum(1 for t in targets if t.sheet_image_path)} screenshots generated")


def step2_analyze_regions(
    targets: List[TargetInfo],
    llm: LLMAdapter,
    max_dimension: int,
    max_retries: int = 3,
) -> None:
    """Step 2: Analyze regions for all sheets using LLM."""
    print("\n" + "=" * 80)
    print("STEP 2: ANALYZING REGIONS")
    print("=" * 80)

    parser = ExcelParser()

    for target in targets:
        skip, reason = should_skip_target(target, "region analysis")
        if skip:
            print(f"  [skip] {target.output_name} ({reason})")
            continue

        if not target.sheet_image_path or not target.sheet_image_path.exists():
            print(f"  [skip] {target.output_name} (no screenshot from step 1)")
            continue

        existing_regions = target.output_dir / "regions" / "output.json"
        if is_file_valid(existing_regions, min_size=10):
            target.region_output_path = existing_regions
            print(f"  [ok] Already processed: {target.output_name}")
            continue

        cleanup_incomplete_file(existing_regions)
        print(f"  Processing: {target.output_name}")

        try:
            # Parse operations from workbook
            all_ops = parser.parse(filepath=str(target.workbook_path))
            sheet_ops = [
                op for op in all_ops
                if op.cell_range.sheet == target.sheet_name
            ]

            # Find bounding boxes
            regions_dict = find_bounding_boxes(sheet_ops, max_sheet_limit=max_dimension)

            # Get merged cells
            merged_cells_list = [
                op.cell_range.range for op in sheet_ops
                if isinstance(op, MergeCells)
            ]

            # Create regions output directory
            regions_dir = target.output_dir / "regions"
            regions_dir.mkdir(parents=True, exist_ok=True)

            # Get sheet range
            sheet_range = f"A1:{chr(64 + min(max_dimension, 26))}{max_dimension}"

            # Analyze regions via LLM
            parsed_output, response, time_taken, prompt, attempts = analyze_sheet_regions(
                llm=llm,
                sheet_image_path=str(target.sheet_image_path),
                sheet_name=target.sheet_name,
                sheet_range=sheet_range,
                regions_dict=regions_dict,
                merged_cells_list=merged_cells_list,
                sheet_operations=sheet_ops,
                max_sheet_limit=max_dimension,
                max_retries=max_retries,
            )

            # Save output.json
            output_json = regions_dir / "output.json"
            if parsed_output:
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(parsed_output.model_dump(), f, indent=2)
                target.region_output_path = output_json
                region_count = len(parsed_output.regions) if parsed_output.regions else 0
                print(f"    [ok] Region analysis complete ({region_count} regions)")
            else:
                print(f"    [warn] Region analysis returned no output")

            # Save raw response for debugging
            if response:
                with open(regions_dir / "raw_response.txt", "w", encoding="utf-8") as f:
                    f.write(response)
            with open(regions_dir / "prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)

        except Exception as e:
            error_msg = f"Region analysis failed: {str(e)}"
            target.errors.append(error_msg)
            print(f"    [error] {error_msg}")
            logger.error(f"Error analyzing regions for {target.output_name}: {e}", exc_info=True)

        # Memory cleanup after each target
        force_garbage_collection(f"after {target.output_name}")

    force_garbage_collection("after step 2 region analysis")
    print(f"\nStep 2 complete: {sum(1 for t in targets if t.region_output_path)} region analyses completed")


def step3_sample_heuristics(
    targets: List[TargetInfo],
    seed: Optional[int] = None,
) -> None:
    """Step 3: Sample heuristics configurations for all targets."""
    print("\n" + "=" * 80)
    print("STEP 3: SAMPLING HEURISTICS")
    print("=" * 80)

    for target in targets:
        skip, reason = should_skip_target(target, "heuristics sampling")
        if skip:
            print(f"  [skip] {target.output_name} ({reason})")
            continue

        existing_config = target.output_dir / "sampled_configs" / "sampled_config.json"
        if existing_config.exists():
            try:
                with open(existing_config, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                if isinstance(config_data, dict) and len(config_data) > 0:
                    target.sampled_config = config_data
                    target.sampled_config_path = existing_config
                    print(f"  [ok] Already processed: {target.output_name}")
                    continue
                else:
                    cleanup_incomplete_file(existing_config)
            except (json.JSONDecodeError, IOError):
                cleanup_incomplete_file(existing_config)

        print(f"  Processing: {target.output_name}")

        try:
            config = sample_config(seed=seed)
            target.sampled_config = config

            config_dir = target.output_dir / "sampled_configs"
            config_dir.mkdir(parents=True, exist_ok=True)

            config_path = config_dir / "sampled_config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            target.sampled_config_path = config_path
            print(f"    [ok] Config sampled and saved")

        except Exception as e:
            error_msg = f"Config sampling failed: {str(e)}"
            target.errors.append(error_msg)
            print(f"    [error] {error_msg}")
            logger.error(f"Error sampling config for {target.output_name}: {e}", exc_info=True)

    print(f"\nStep 3 complete: {sum(1 for t in targets if t.sampled_config)} configs sampled")


def step4_generate_sequences(
    targets: List[TargetInfo],
    max_dimension: int,
) -> None:
    """Step 4: Generate sequences using sampled heuristics."""
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING SEQUENCES")
    print("=" * 80)

    parser = ExcelParser()

    for target in targets:
        skip, reason = should_skip_target(target, "sequence generation")
        if skip:
            print(f"  [skip] {target.output_name} ({reason})")
            continue

        if not target.sampled_config:
            print(f"  [skip] {target.output_name} (no sampled config from step 3)")
            continue

        existing_sequence = target.output_dir / "sequences" / "framework_output.txt"
        if is_file_valid(existing_sequence, min_size=10):
            target.sequence_output_path = target.output_dir / "sequences"
            target.framework_output_path = existing_sequence
            print(f"  [ok] Already processed: {target.output_name}")
            continue

        cleanup_incomplete_file(existing_sequence)
        print(f"  Processing: {target.output_name}")

        try:
            # Load operations
            all_ops = parser.parse(filepath=str(target.workbook_path))
            sheet_ops = [
                op for op in all_ops
                if op.cell_range.sheet == target.sheet_name
            ]

            # Filter by dimension
            filtered_ops = []
            for op in sheet_ops:
                _, _, end_row, end_col = op.cell_range.get_coordinates()
                if end_row <= max_dimension and end_col <= max_dimension:
                    filtered_ops.append(op)

            if not filtered_ops:
                print(f"    [warn] No operations found within dimension limits")
                continue

            # Load region metadata if available
            region_metadata = None
            if target.region_output_path and target.region_output_path.exists():
                with open(target.region_output_path, "r", encoding="utf-8") as f:
                    region_metadata = json.load(f)

            # Create sequencing engine with sampled config
            engine = SequencingEngine(target.sampled_config)

            # Sequence operations
            sequenced_ops = engine.sequence(
                operations=list(filtered_ops),
                region_metadata=region_metadata,
                sheet_name=target.sheet_name,
            )

            # Convert to symbolic
            symbolic_lines = operations_to_symbolic(sequenced_ops)

            # Save outputs
            sequences_dir = target.output_dir / "sequences"
            sequences_dir.mkdir(parents=True, exist_ok=True)

            framework_output_path = sequences_dir / "framework_output.txt"
            with open(framework_output_path, "w", encoding="utf-8") as f:
                for line in symbolic_lines:
                    f.write(line + "\n")

            target.sequence_output_path = sequences_dir
            target.framework_output_path = framework_output_path

            print(f"    [ok] Sequence generated ({len(symbolic_lines)} operations)")

        except Exception as e:
            error_msg = f"Sequence generation failed: {str(e)}"
            target.errors.append(error_msg)
            print(f"    [error] {error_msg}")
            logger.error(f"Error generating sequence for {target.output_name}: {e}", exc_info=True)

    force_garbage_collection("after step 4 sequence generation")
    print(f"\nStep 4 complete: {sum(1 for t in targets if t.framework_output_path)} sequences generated")


def step5_run_refinement(
    targets: List[TargetInfo],
    max_dimension: int,
    model_name: str,
    provider: str,
    cache_dir: Path,
    use_cache: bool,
    max_iterations: int = 3,
    max_retries: int = 3,
    temperature: float = 0.15,
    reasoning_effort: str = "low",
    max_completion_tokens: Optional[int] = None,
    judge_reasoning_effort: str = "high",
    judge_max_completion_tokens: Optional[int] = 20000,
) -> None:
    """Step 5: Run sequence refinement for all targets."""
    print("\n" + "=" * 80)
    print("STEP 5: RUNNING SEQUENCE REFINEMENT")
    print("=" * 80)

    cache_path = cache_dir / "refinement_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    for target in targets:
        skip, reason = should_skip_target(target, "refinement")
        if skip:
            print(f"  [skip] {target.output_name} ({reason})")
            continue

        if not target.framework_output_path or not target.framework_output_path.exists():
            print(f"  [skip] {target.output_name} (no framework output from step 4)")
            continue

        # Check if refinement already completed
        refinement_dir = target.output_dir / "refinement"
        if refinement_dir.exists():
            summary_file = refinement_dir / "summary.json"
            if is_file_valid(summary_file, min_size=10):
                try:
                    with open(summary_file, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    target.refinement_output_dir = refinement_dir
                    target.refinement_success = summary.get("success", False)
                    status = "success" if target.refinement_success else "failed"
                    print(f"  [ok] Already processed: {target.output_name} (refinement {status})")
                    continue
                except (json.JSONDecodeError, IOError):
                    cleanup_incomplete_file(summary_file)

        print(f"  Processing: {target.output_name}")

        try:
            refinement_dir.mkdir(parents=True, exist_ok=True)
            capture_dir = refinement_dir / "images"

            config = RefinementConfig(
                step_file=target.framework_output_path,
                final_workbook=target.workbook_path,
                sheet_name=target.sheet_name,
                output_dir=refinement_dir,
                sheet_image_path=target.sheet_image_path,
                allow_image_capture=True,
                capture_dir=capture_dir,
                max_dimension=max_dimension,
                max_iterations=max_iterations,
                max_retries=max_retries,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
                provider=provider,
                model=model_name,
                cache_path=cache_path,
                use_cache=use_cache,
                log_progress=True,
                judge_provider=provider,
                judge_model=model_name,
                judge_reasoning_effort=judge_reasoning_effort,
                judge_max_completion_tokens=judge_max_completion_tokens,
                judge_cache_path=cache_path,
                judge_use_cache=use_cache,
            )

            pipeline = SequenceRefinementPipeline(config)
            outcome = pipeline.run()

            target.refinement_output_dir = refinement_dir
            target.refinement_success = outcome.success

            if outcome.success:
                print(f"    [ok] Refinement successful (iterations: {outcome.iterations})")

                if outcome.best_operations:
                    best_ops_path = refinement_dir / "best_operations.txt"
                    with open(best_ops_path, "w", encoding="utf-8") as f:
                        for op in outcome.best_operations:
                            f.write(op + "\n")
            else:
                print(f"    [warn] Refinement failed: {outcome.message}")

        except Exception as e:
            error_msg = f"Refinement failed: {str(e)}"
            target.errors.append(error_msg)
            print(f"    [error] {error_msg}")
            logger.error(f"Error running refinement for {target.output_name}: {e}", exc_info=True)

        # Critical: clean up after each refinement
        pipeline = None
        config = None
        force_garbage_collection(f"after refinement {target.output_name}")

    force_garbage_collection("after step 5 refinement")
    print(f"\nStep 5 complete: {sum(1 for t in targets if t.refinement_output_dir)} refinements completed")


def step6_generate_gifs(
    targets: List[TargetInfo],
    output_dir: Path,
    gif_duration: int = 300,
    gif_loop: int = 1,
    max_op_len: Optional[int] = None,
) -> None:
    """Step 6: Generate GIFs from refined or original sequences.

    Uses the code generator from next_action_pred_eval to create xlwings
    code that takes snapshots, then combines them into a GIF.
    """
    print("\n" + "=" * 80)
    print("STEP 6: GENERATING GIFS")
    print("=" * 80)

    try:
        from next_action_pred_eval.utils.codegen.code_generator import OfficeJSGenerator
    except ImportError:
        OfficeJSGenerator = None

    skipped_due_to_op_len = []

    for target in targets:
        if target.skipped:
            print(f"  [skip] {target.output_name} (already marked as skipped)")
            continue
        if target.errors:
            print(f"  [skip] {target.output_name} (has errors)")
            continue
        if target.refinement_success is False:
            print(f"  [skip] {target.output_name} (refinement failed)")
            continue
        if target.refinement_success is not True:
            print(f"  [skip] {target.output_name} (refinement not completed)")
            continue

        gif_output_dir = target.refinement_output_dir

        existing_gif = gif_output_dir / "operation_sequence.gif"
        if is_gif_valid(existing_gif):
            target.gif_path = existing_gif
            print(f"  [ok] Already processed: {target.output_name}")
            continue

        cleanup_incomplete_file(existing_gif)

        # Find best_operations.txt
        best_ops_path = target.refinement_output_dir / "best_operations.txt"
        if not best_ops_path.exists():
            run_dirs = list(target.refinement_output_dir.glob("run_*"))
            for run_dir in sorted(run_dirs, reverse=True):
                candidate = run_dir / "best_operations.txt"
                if candidate.exists():
                    best_ops_path = candidate
                    break

        if not best_ops_path.exists():
            print(f"  [skip] {target.output_name} (no best_operations.txt found)")
            continue

        print(f"  Processing: {target.output_name}")

        try:
            with open(best_ops_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            ops = symbolic_to_operations(lines)

            if not ops:
                print(f"    [warn] No operations found in {best_ops_path.name}")
                continue

            if max_op_len is not None and len(ops) > max_op_len:
                print(f"    [skip] {len(ops)} operations exceeds limit of {max_op_len}")
                skipped_due_to_op_len.append({
                    "target": target.output_name,
                    "workbook_path": str(target.workbook_path),
                    "sheet_name": target.sheet_name,
                    "operation_count": len(ops),
                    "limit": max_op_len,
                })
                continue

            # Create snapshots directory
            snapshots_dir = gif_output_dir / "snapshots_dir"
            if snapshots_dir.exists():
                shutil.rmtree(snapshots_dir)
            snapshots_dir.mkdir(parents=True, exist_ok=True)

            # Generate snapshot images via xlwings
            _generate_snapshots_xlwings(ops, target, snapshots_dir)

            # Sort snapshot files
            def sort_key(path: Path) -> tuple:
                stem = path.stem
                if "_step_" in stem:
                    sheet_part, _, step_part = stem.rpartition("_step_")
                    try:
                        return (sheet_part, int(step_part))
                    except ValueError:
                        pass
                return (stem, 0)

            image_files = sorted(snapshots_dir.glob("*.png"), key=sort_key)

            if not image_files:
                print(f"    [warn] No snapshot images generated")
                continue

            # Create GIF
            gif_path = gif_output_dir / "operation_sequence.gif"
            create_gif_from_images(
                [str(p) for p in image_files],
                str(gif_path),
                duration=gif_duration,
                loop=gif_loop,
            )

            target.gif_path = gif_path
            print(f"    [ok] GIF generated ({len(image_files)} frames)")

            # Clean up snapshot images
            try:
                if snapshots_dir.exists():
                    shutil.rmtree(snapshots_dir)
            except Exception:
                pass

        except Exception as e:
            error_msg = f"GIF generation failed: {str(e)}"
            target.errors.append(error_msg)
            print(f"    [error] {error_msg}")
            logger.error(f"Error generating GIF for {target.output_name}: {e}", exc_info=True)

        force_garbage_collection(f"after GIF {target.output_name}")

    # Save list of targets skipped due to operation length limit
    if skipped_due_to_op_len:
        skipped_file = output_dir / "gif_skipped_due_to_op_len.json"
        with open(skipped_file, "w", encoding="utf-8") as f:
            json.dump(skipped_due_to_op_len, f, indent=2)
        print(f"\n  Saved {len(skipped_due_to_op_len)} skipped targets to: {skipped_file}")

    force_garbage_collection("after step 6 GIF generation")
    print(f"\nStep 6 complete: {sum(1 for t in targets if t.gif_path)} GIFs generated")


def _generate_snapshots_xlwings(
    ops: List,
    target: TargetInfo,
    snapshots_dir: Path,
) -> None:
    """Generate snapshot images by applying operations step-by-step via xlwings.

    Applies each operation using the StateBuilder, writes the intermediate
    state to a temporary workbook, and captures a screenshot at each step.
    This avoids ``exec()`` and keeps Excel process management explicit.
    """
    try:
        import xlwings as xw
    except ImportError:
        raise RuntimeError("xlwings is required for GIF snapshot generation")

    from next_action_pred_eval.core.state import StateBuilder
    from next_action_pred_eval.utils.workbook.state_to_sheet import state_to_workbook

    builder = StateBuilder()
    app = xw.App(visible=False, add_book=False)
    try:
        for step_idx, op in enumerate(ops):
            builder.apply(op)

            # Write current state to a temp workbook and capture image
            state = builder.to_state()
            wb = app.books.add()
            try:
                state_to_workbook(state, wb)
                sht = wb.sheets[target.sheet_name] if target.sheet_name in [s.name for s in wb.sheets] else wb.sheets[0]
                used = sht.used_range
                if used is not None:
                    capture_range = sht.range(
                        (1, 1),
                        (min(used.last_cell.row, 200), min(used.last_cell.column, 200)),
                    )
                    img_path = snapshots_dir / f"{target.sheet_name}_step_{step_idx:04d}.png"
                    capture_range.to_png(str(img_path))
            finally:
                wb.close()
    finally:
        app.quit()


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: Path) -> Path:
    """Set up logging to both console and file, and redirect print() to log file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"pipeline_log_{timestamp}.txt"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    class TeeOutput:
        """Tee stdout to both console and log file."""
        def __init__(self, log_path: Path):
            self.terminal = sys.__stdout__
            self.log = open(log_path, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            if message.strip():
                self.log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
            else:
                self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = TeeOutput(log_file)

    return log_file


# =============================================================================
# TARGET LOADING
# =============================================================================

def load_workbook_targets(json_file_path: str) -> List[Dict[str, str]]:
    """Load workbook targets from a JSON file.

    Expected format: list of objects with ``workbook_path`` and ``sheet_name``.
    """
    json_path = Path(json_file_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    targets = []
    for entry in data:
        targets.append({
            "workbook_path": entry["workbook_path"],
            "sheet_name": entry["sheet_name"],
        })
    return targets


# =============================================================================
# RESULT REPORTING
# =============================================================================

def organize_unsuccessful_targets(targets: List[TargetInfo], output_dir: Path) -> None:
    """Move unsuccessful target folders to organized locations."""
    errors_dir = output_dir / "unsuccessful" / "errors"
    failed_refinement_dir = output_dir / "unsuccessful" / "failed_refinement"

    moved_errors = 0
    moved_failed = 0

    for target in targets:
        if target.skipped:
            continue
        if not target.output_dir.exists():
            continue
        if "unsuccessful" in str(target.output_dir):
            continue

        if target.errors:
            errors_dir.mkdir(parents=True, exist_ok=True)
            dest = errors_dir / target.output_name
            try:
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(target.output_dir), str(dest))
                target.output_dir = dest
                moved_errors += 1
            except Exception as e:
                logger.warning(f"Failed to move {target.output_name} to errors folder: {e}")

        elif target.refinement_success is False:
            failed_refinement_dir.mkdir(parents=True, exist_ok=True)
            dest = failed_refinement_dir / target.output_name
            try:
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(target.output_dir), str(dest))
                target.output_dir = dest
                moved_failed += 1
            except Exception as e:
                logger.warning(f"Failed to move {target.output_name} to failed_refinement folder: {e}")

    if moved_errors > 0 or moved_failed > 0:
        print(f"\nOrganized unsuccessful targets:")
        if moved_errors > 0:
            print(f"  - Moved {moved_errors} target(s) with errors to: {errors_dir}")
        if moved_failed > 0:
            print(f"  - Moved {moved_failed} target(s) with failed refinement to: {failed_refinement_dir}")

    # Save unsuccessful_runs.json
    unsuccessful_runs = []
    for target in targets:
        if target.skipped:
            continue
        if target.errors:
            unsuccessful_runs.append({
                "workbook_path": str(target.workbook_path),
                "sheet_name": target.sheet_name,
                "status": "error",
                "errors": target.errors,
            })
        elif target.refinement_success is False:
            unsuccessful_runs.append({
                "workbook_path": str(target.workbook_path),
                "sheet_name": target.sheet_name,
                "status": "failed_refinement",
            })

    if unsuccessful_runs:
        unsuccessful_dir = output_dir / "unsuccessful"
        unsuccessful_dir.mkdir(parents=True, exist_ok=True)
        unsuccessful_json_path = unsuccessful_dir / "unsuccessful_runs.json"
        with open(unsuccessful_json_path, "w", encoding="utf-8") as f:
            json.dump(unsuccessful_runs, f, indent=4)


def save_pipeline_summary(result: PipelineResult, output_dir: Path) -> None:
    """Save pipeline summary to file."""
    summary_path = output_dir / "pipeline_summary.json"

    summary = {
        "total_targets": result.total_targets,
        "successful": result.successful,
        "failed": result.failed,
        "skipped": result.skipped,
        "duration_seconds": result.duration_seconds,
        "start_time": result.start_time.isoformat(),
        "end_time": result.end_time.isoformat(),
        "skipped_list": result.skipped_list,
        "errors": result.errors,
        "targets": [
            {
                "name": t.output_name,
                "workbook": str(t.workbook_path),
                "sheet": t.sheet_name,
                "skipped": t.skipped,
                "skip_reason": t.skip_reason,
                "has_screenshot": t.sheet_image_path is not None,
                "has_regions": t.region_output_path is not None,
                "has_config": t.sampled_config_path is not None,
                "has_sequence": t.framework_output_path is not None,
                "has_refinement": t.refinement_output_dir is not None,
                "refinement_success": t.refinement_success,
                "has_gif": t.gif_path is not None,
                "gif_path": str(t.gif_path) if t.gif_path else None,
                "errors": t.errors,
            }
            for t in result.targets
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if result.errors:
        errors_path = output_dir / "pipeline_errors.json"
        with open(errors_path, "w", encoding="utf-8") as f:
            json.dump(result.errors, f, indent=2)

    if result.skipped_list:
        skipped_path = output_dir / "pipeline_skipped.json"
        with open(skipped_path, "w", encoding="utf-8") as f:
            json.dump(result.skipped_list, f, indent=2)

    # Save refinement failures
    refinement_failures = [
        {
            "target": t.output_name,
            "workbook": str(t.workbook_path),
            "sheet": t.sheet_name,
            "gif_path": str(t.gif_path) if t.gif_path else None,
        }
        for t in result.targets
        if t.refinement_success is False
    ]
    if refinement_failures:
        failures_path = output_dir / "refinement_failures.json"
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump(refinement_failures, f, indent=2)

    # Step-by-step failure breakdown
    step_failures = {
        "screenshot_failures": [
            {"target": t.output_name, "errors": t.errors}
            for t in result.targets
            if not t.skipped and not t.sheet_image_path and t.errors
        ],
        "region_analysis_failures": [
            {"target": t.output_name, "errors": t.errors}
            for t in result.targets
            if t.sheet_image_path and not t.region_output_path and t.errors
        ],
        "sequence_generation_failures": [
            {"target": t.output_name, "errors": t.errors}
            for t in result.targets
            if t.sampled_config and not t.framework_output_path and t.errors
        ],
        "gif_generation_failures": [
            {"target": t.output_name, "errors": t.errors}
            for t in result.targets
            if t.framework_output_path and not t.gif_path and t.errors
        ],
    }
    if any(step_failures.values()):
        step_failures_path = output_dir / "step_failures.json"
        with open(step_failures_path, "w", encoding="utf-8") as f:
            json.dump(step_failures, f, indent=2)

    # Save successful targets list
    successful_targets = [
        {
            "target": t.output_name,
            "workbook": str(t.workbook_path),
            "sheet": t.sheet_name,
            "refinement_success": t.refinement_success,
            "gif_path": str(t.gif_path) if t.gif_path else None,
        }
        for t in result.targets
        if not t.skipped and not t.errors and t.gif_path
    ]
    if successful_targets:
        successful_path = output_dir / "successful_targets.json"
        with open(successful_path, "w", encoding="utf-8") as f:
            json.dump(successful_targets, f, indent=2)


def print_final_summary(result: PipelineResult, output_dir: Path) -> None:
    """Print final pipeline summary."""
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Duration: {result.duration_seconds:.2f} seconds")
    print(f"\nResults:")
    print(f"  Total targets:  {result.total_targets}")
    print(f"  Successful:     {result.successful}")
    print(f"  Failed:         {result.failed}")
    print(f"  Skipped:        {result.skipped}")

    refinement_failures = [t for t in result.targets if t.refinement_success is False]
    refinement_successes = [t for t in result.targets if t.refinement_success is True]
    print(f"\nRefinement results:")
    print(f"  Successful:     {len(refinement_successes)}")
    print(f"  Failed:         {len(refinement_failures)}")

    if refinement_failures:
        print(f"\nRefinement failures ({len(refinement_failures)}):")
        for t in refinement_failures:
            gif_status = "GIF created from original sequence" if t.gif_path else "No GIF"
            print(f"  - {t.output_name}: {gif_status}")

    if result.skipped_list:
        print(f"\nSkipped targets ({len(result.skipped_list)}):")
        for item in result.skipped_list:
            print(f"  - {item['target']}: {item['reason']}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for item in result.errors:
            print(f"  - {item['target']}:")
            for err in item["errors"]:
                print(f"      {err}")

    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - pipeline_summary.json (complete summary)")
    print(f"  - pipeline_log_*.txt (full execution log)")
    if result.errors:
        print(f"  - pipeline_errors.json (all errors)")
    if result.skipped_list:
        print(f"  - pipeline_skipped.json (skipped targets)")
    if refinement_failures:
        print(f"  - refinement_failures.json (refinement failures)")

    step_failures_path = output_dir / "step_failures.json"
    if step_failures_path.exists():
        print(f"  - step_failures.json (failures by step)")

    successful_path = output_dir / "successful_targets.json"
    if successful_path.exists():
        print(f"  - successful_targets.json (successful targets)")

    print("=" * 80)


# =============================================================================
# LLM ADAPTER FACTORY
# =============================================================================

def create_llm_adapter(
    provider: str,
    model: str,
    cache_dir: Path,
    use_cache: bool,
) -> LLMAdapter:
    """Create an LLM adapter instance based on provider and model name.

    This factory imports the appropriate adapter class from
    next_action_pred_eval.utils.llm and configures it.

    Args:
        provider: One of ``"openai"`` or ``"azure"``.
        model: Model name / deployment name.
        cache_dir: Directory for caching LLM responses.
        use_cache: Whether to enable caching.

    Returns:
        An initialized LLMAdapter instance.
    """
    cache_file = str(cache_dir / "region_analysis_cache.json") if use_cache else None
    cache_dir.mkdir(parents=True, exist_ok=True)

    if provider in ("openai", "azure"):
        from next_action_pred_eval.utils.llm.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(
            model=model,
            cache_enabled=use_cache,
            cache_path=cache_file,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Supported: 'openai', 'azure'."
        )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    workbook_targets: List[Dict[str, str]],
    output_dir: Path,
    llm: LLMAdapter,
    provider: str = "openai",
    model_name: str = "gpt-4o",
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = DEFAULT_USE_CACHE,
    region_max_retries: int = DEFAULT_REGION_ANALYSIS_MAX_RETRIES,
    config_sampler_seed: Optional[int] = DEFAULT_CONFIG_SAMPLER_SEED,
    refinement_max_iterations: int = DEFAULT_REFINEMENT_MAX_ITERATIONS,
    refinement_max_retries: int = DEFAULT_REFINEMENT_MAX_RETRIES,
    refinement_temperature: float = DEFAULT_REFINEMENT_TEMPERATURE,
    refinement_reasoning_effort: str = DEFAULT_REFINEMENT_REASONING_EFFORT,
    refinement_max_completion_tokens: Optional[int] = DEFAULT_REFINEMENT_MAX_COMPLETION_TOKENS,
    judge_reasoning_effort: str = DEFAULT_JUDGE_REASONING_EFFORT,
    judge_max_completion_tokens: Optional[int] = DEFAULT_JUDGE_MAX_COMPLETION_TOKENS,
    gif_duration: int = DEFAULT_GIF_DURATION_MS,
    gif_loop: int = DEFAULT_GIF_LOOP,
    max_op_len_for_gif: Optional[int] = DEFAULT_MAX_OP_LEN_FOR_GIF,
    stages: Optional[Dict[str, bool]] = None,
) -> PipelineResult:
    """Run the full 6-step pipeline for all targets."""
    start_time = datetime.now()

    log_file = setup_logging(output_dir)

    print("=" * 80)
    print("FULL PIPELINE EXECUTION")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    print("=" * 80)

    # Initialize targets
    targets: List[TargetInfo] = []
    skipped_list: List[Dict[str, str]] = []

    print("\nInitializing targets...")

    for entry in workbook_targets:
        workbook_path = Path(entry["workbook_path"])
        sheet_name = entry["sheet_name"]
        output_name = get_output_name(workbook_path, sheet_name)
        target_output_dir = output_dir / output_name

        target = TargetInfo(
            workbook_path=workbook_path,
            sheet_name=sheet_name,
            output_name=output_name,
            output_dir=target_output_dir,
        )

        if not workbook_path.exists():
            target.skipped = True
            target.skip_reason = f"Workbook not found: {workbook_path}"
            skipped_list.append({
                "target": output_name,
                "reason": target.skip_reason,
            })
            print(f"  [skip] {output_name}: Workbook not found")
            targets.append(target)
            continue

        print(f"  [ok] {output_name}: Ready for processing")
        targets.append(target)

    # Restore state from previous runs
    print("\nRestoring state from previous runs...")
    restored_count = 0
    for target in targets:
        if not target.skipped and target.output_dir.exists():
            restore_target_state(target)
            steps_restored = sum([
                target.sheet_image_path is not None,
                target.region_output_path is not None,
                target.sampled_config is not None,
                target.framework_output_path is not None,
                target.refinement_output_dir is not None,
                target.gif_path is not None,
            ])
            if steps_restored > 0:
                restored_count += 1

    if restored_count > 0:
        print(f"  Restored state for {restored_count} targets from previous runs")
    else:
        print("  No previous state found")

    fully_complete = [t for t in targets if is_target_fully_complete(t)]
    print(f"  {len(fully_complete)} targets already fully complete")

    output_dir.mkdir(parents=True, exist_ok=True)

    active_targets = [t for t in targets if not t.skipped]
    needs_processing = [t for t in active_targets if not is_target_fully_complete(t)]
    print(
        f"\n{len(active_targets)} targets ready for processing, "
        f"{len(skipped_list)} skipped, {len(needs_processing)} need processing"
    )

    if active_targets:
        # Stage toggles (default all enabled)
        stages = stages or {}
        run_screenshots = stages.get("screenshots", True)
        run_regions = stages.get("regions", True)
        run_sequencing = stages.get("sequencing", True)
        run_refinement_stage = stages.get("refinement", True)
        run_gif = stages.get("gif", True)

        # Step 1: Generate screenshots
        if run_screenshots:
            step1_generate_screenshots(targets, max_dimension)
        else:
            print("\n[SKIP] Step 1: Screenshots (disabled in config)")

        # Step 2: Analyze regions
        if run_regions:
            step2_analyze_regions(targets, llm, max_dimension, region_max_retries)
        else:
            print("\n[SKIP] Step 2: Region analysis (disabled in config)")

        # Step 3+4: Sample heuristics + Generate sequences
        if run_sequencing:
            step3_sample_heuristics(targets, config_sampler_seed)
            step4_generate_sequences(targets, max_dimension)
        else:
            print("\n[SKIP] Step 3+4: Sequencing (disabled in config)")

        # Step 5: Run refinement
        if run_refinement_stage:
            step5_run_refinement(
                targets,
                max_dimension,
                model_name=model_name,
                provider=provider,
                cache_dir=cache_dir,
                use_cache=use_cache,
                max_iterations=refinement_max_iterations,
                max_retries=refinement_max_retries,
                temperature=refinement_temperature,
                reasoning_effort=refinement_reasoning_effort,
                max_completion_tokens=refinement_max_completion_tokens,
                judge_reasoning_effort=judge_reasoning_effort,
                judge_max_completion_tokens=judge_max_completion_tokens,
            )
        else:
            print("\n[SKIP] Step 5: Refinement (disabled in config)")

        # Step 6: Generate GIFs
        if run_gif:
            step6_generate_gifs(targets, output_dir, gif_duration, gif_loop, max_op_len_for_gif)
        else:
            print("\n[SKIP] Step 6: GIF generation (disabled in config)")

    end_time = datetime.now()

    # Collect errors
    all_errors = []
    for target in targets:
        if target.errors:
            all_errors.append({
                "target": target.output_name,
                "errors": target.errors,
            })

    successful = sum(1 for t in targets if not t.skipped and not t.errors and t.gif_path)
    failed = sum(1 for t in targets if not t.skipped and (t.errors or not t.gif_path))
    skipped = sum(1 for t in targets if t.skipped)

    result = PipelineResult(
        total_targets=len(targets),
        successful=successful,
        failed=failed,
        skipped=skipped,
        targets=targets,
        errors=all_errors,
        skipped_list=skipped_list,
        start_time=start_time,
        end_time=end_time,
    )

    organize_unsuccessful_targets(targets, output_dir)
    save_pipeline_summary(result, output_dir)
    print_final_summary(result, output_dir)

    return result


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def load_pipeline_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load pipeline configuration from a YAML file.

    Returns a flat dict with all configuration values, using defaults
    for any keys not specified in the file.
    """
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    llm_cfg = raw.get("llm", {})
    region_cfg = raw.get("region_analysis", {})
    seq_cfg = raw.get("sequencing", {})
    ref_cfg = raw.get("refinement", {})
    gif_cfg = raw.get("gif", {})
    stages_cfg = raw.get("stages", {})

    # Load targets from config or external file
    targets = raw.get("targets", [])
    targets_file = raw.get("targets_file")
    if targets_file and not targets:
        targets = load_workbook_targets(targets_file)

    return {
        "targets": targets,
        "output_dir": raw.get("output_dir", str(DEFAULT_OUTPUT_DIR)),
        # Stages
        "stages": {
            "screenshots": stages_cfg.get("screenshots", True),
            "regions": stages_cfg.get("regions", True),
            "sequencing": stages_cfg.get("sequencing", True),
            "refinement": stages_cfg.get("refinement", True),
            "gif": stages_cfg.get("gif", True),
        },
        # LLM
        "provider": llm_cfg.get("provider", "openai"),
        "model": llm_cfg.get("model", "gpt-4o"),
        "cache_enabled": llm_cfg.get("cache_enabled", DEFAULT_USE_CACHE),
        "cache_dir": llm_cfg.get("cache_dir", str(DEFAULT_CACHE_DIR)),
        # Region analysis
        "max_dimension": region_cfg.get("max_dimension", DEFAULT_MAX_DIMENSION),
        "region_max_retries": region_cfg.get("max_retries", DEFAULT_REGION_ANALYSIS_MAX_RETRIES),
        # Sequencing
        "config_sampler_seed": seq_cfg.get("config_sampler_seed", DEFAULT_CONFIG_SAMPLER_SEED),
        # Refinement
        "refinement_max_iterations": ref_cfg.get("max_iterations", DEFAULT_REFINEMENT_MAX_ITERATIONS),
        "refinement_max_retries": ref_cfg.get("max_retries", DEFAULT_REFINEMENT_MAX_RETRIES),
        "refinement_temperature": ref_cfg.get("temperature", DEFAULT_REFINEMENT_TEMPERATURE),
        "refinement_reasoning_effort": ref_cfg.get("reasoning_effort", DEFAULT_REFINEMENT_REASONING_EFFORT),
        "refinement_max_completion_tokens": ref_cfg.get("max_completion_tokens", DEFAULT_REFINEMENT_MAX_COMPLETION_TOKENS),
        "judge_reasoning_effort": ref_cfg.get("judge_reasoning_effort", DEFAULT_JUDGE_REASONING_EFFORT),
        "judge_max_completion_tokens": ref_cfg.get("judge_max_completion_tokens", DEFAULT_JUDGE_MAX_COMPLETION_TOKENS),
        # GIF
        "gif_duration": gif_cfg.get("duration_ms", DEFAULT_GIF_DURATION_MS),
        "gif_loop": gif_cfg.get("loop", DEFAULT_GIF_LOOP),
        "max_op_len_for_gif": gif_cfg.get("max_operations", DEFAULT_MAX_OP_LEN_FOR_GIF),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate training data via a configurable pipeline.\n"
                    "Primary usage: --config pipeline.yaml\n"
                    "CLI args override config file values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to pipeline config YAML (recommended). "
             "See configs/generation/pipeline.yaml for template.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        help="Path to JSON file with workbook targets (overrides config).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (overrides config).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "azure"],
        help="LLM provider (overrides config).",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (overrides config).",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=["screenshots", "regions", "sequencing", "refinement", "gif"],
        default=[],
        help="Stages to skip (overrides config).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["screenshots", "regions", "sequencing", "refinement", "gif"],
        help="Run ONLY these stages (overrides config and --skip).",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM caching.")
    parser.add_argument("--config-seed", type=int, help="Sequencing config sampler seed.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")

    return parser.parse_args()


def main():
    """Main entry point for CLI invocation."""
    args = parse_args()

    # Load config from YAML if provided, otherwise use defaults
    if args.config:
        cfg = load_pipeline_config(args.config)
    else:
        cfg = {
            "targets": [],
            "output_dir": str(DEFAULT_OUTPUT_DIR),
            "stages": {},
            "provider": "openai",
            "model": "gpt-4o",
            "cache_enabled": DEFAULT_USE_CACHE,
            "cache_dir": str(DEFAULT_CACHE_DIR),
            "max_dimension": DEFAULT_MAX_DIMENSION,
            "region_max_retries": DEFAULT_REGION_ANALYSIS_MAX_RETRIES,
            "config_sampler_seed": DEFAULT_CONFIG_SAMPLER_SEED,
            "refinement_max_iterations": DEFAULT_REFINEMENT_MAX_ITERATIONS,
            "refinement_max_retries": DEFAULT_REFINEMENT_MAX_RETRIES,
            "refinement_temperature": DEFAULT_REFINEMENT_TEMPERATURE,
            "refinement_reasoning_effort": DEFAULT_REFINEMENT_REASONING_EFFORT,
            "refinement_max_completion_tokens": DEFAULT_REFINEMENT_MAX_COMPLETION_TOKENS,
            "judge_reasoning_effort": DEFAULT_JUDGE_REASONING_EFFORT,
            "judge_max_completion_tokens": DEFAULT_JUDGE_MAX_COMPLETION_TOKENS,
            "gif_duration": DEFAULT_GIF_DURATION_MS,
            "gif_loop": DEFAULT_GIF_LOOP,
            "max_op_len_for_gif": DEFAULT_MAX_OP_LEN_FOR_GIF,
        }

    # CLI overrides
    if args.targets:
        cfg["targets"] = load_workbook_targets(args.targets)
    if args.output:
        cfg["output_dir"] = args.output
    if args.provider:
        cfg["provider"] = args.provider
    if args.model:
        cfg["model"] = args.model
    if args.no_cache:
        cfg["cache_enabled"] = False
    if args.config_seed is not None:
        cfg["config_sampler_seed"] = args.config_seed

    # Stage toggles: --only takes precedence over --skip
    stages = dict(cfg.get("stages", {}))
    if args.only:
        all_stages = ["screenshots", "regions", "sequencing", "refinement", "gif"]
        stages = {s: (s in args.only) for s in all_stages}
    elif args.skip:
        for s in args.skip:
            stages[s] = False

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate targets
    workbook_targets = cfg["targets"]
    if not workbook_targets:
        print("ERROR: No targets specified. Use --config with targets or --targets.")
        sys.exit(1)

    output_dir = Path(cfg["output_dir"])
    cache_dir = Path(cfg["cache_dir"])
    use_cache = cfg["cache_enabled"]
    max_op_len = cfg["max_op_len_for_gif"]
    if max_op_len and max_op_len <= 0:
        max_op_len = None

    # Print stage plan
    print("Pipeline stages:")
    for stage_name in ["screenshots", "regions", "sequencing", "refinement", "gif"]:
        enabled = stages.get(stage_name, True)
        print(f"  {stage_name:15s} {'ON' if enabled else 'SKIP'}")
    print()

    # Create LLM adapter
    llm = create_llm_adapter(
        provider=cfg["provider"],
        model=cfg["model"],
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    # Run pipeline
    result = run_pipeline(
        workbook_targets=workbook_targets,
        output_dir=output_dir,
        llm=llm,
        provider=cfg["provider"],
        model_name=cfg["model"],
        max_dimension=cfg["max_dimension"],
        cache_dir=cache_dir,
        use_cache=use_cache,
        region_max_retries=cfg["region_max_retries"],
        config_sampler_seed=cfg["config_sampler_seed"],
        refinement_max_iterations=cfg["refinement_max_iterations"],
        refinement_max_retries=cfg["refinement_max_retries"],
        refinement_temperature=cfg["refinement_temperature"],
        refinement_reasoning_effort=cfg["refinement_reasoning_effort"],
        refinement_max_completion_tokens=cfg["refinement_max_completion_tokens"],
        judge_reasoning_effort=cfg["judge_reasoning_effort"],
        judge_max_completion_tokens=cfg["judge_max_completion_tokens"],
        gif_duration=cfg["gif_duration"],
        gif_loop=cfg["gif_loop"],
        max_op_len_for_gif=max_op_len,
        stages=stages,
    )

    # Exit with non-zero if there were failures
    if result.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
