#!/usr/bin/env python
"""
Convert old trajectory files to new JSON format.

Usage:
    # Default: Convert all trajectories from data/raw (uses dataset.jsonl)
    python scripts/convert_trajectories.py

    # Convert from data/raw with max files limit
    python scripts/convert_trajectories.py --max-files 16

    # Convert from external JSONL config file
    python scripts/convert_trajectories.py \
        --jsonl path/to/operations_index.jsonl \
        --base-dir path/to/operations_root \
        --output-dir data/trajectories

    # Convert single file
    python scripts/convert_trajectories.py \
        --input path/to/operations.txt \
        --output data/trajectories/trajectory.json \
        --label my_trajectory
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def load_operations_from_txt(path: Path) -> List[str]:
    """Load symbolic operations from a text file."""
    operations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                operations.append(line)
    return operations


def convert_single_file(
    input_path: Path,
    output_path: Path,
    label: str,
    metadata: Optional[dict] = None,
) -> bool:
    """Convert a single txt file to JSON format."""
    try:
        operations = load_operations_from_txt(input_path)

        if not operations:
            print(f"  WARNING: No operations found in {input_path}")
            return False

        trajectory = {
            "name": label,
            "description": f"Converted from {input_path.name}",
            "operations": operations,
            "metadata": metadata or {
                "source": str(input_path),
                "original_label": label,
            }
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trajectory, f, indent=2)

        print(f"  OK: {label} -> {output_path.name} ({len(operations)} ops)")
        return True

    except Exception as e:
        print(f"  ERROR: {label}: {e}")
        return False


def convert_from_jsonl(
    jsonl_path: Path,
    base_dir: Path,
    output_dir: Path,
    max_files: Optional[int] = None,
) -> tuple:
    """Convert trajectories listed in a JSONL config file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if max_files:
        entries = entries[:max_files]

    success_count = 0
    fail_count = 0

    print(f"Converting {len(entries)} trajectories from {jsonl_path.name}")
    print(f"Base dir: {base_dir}")
    print(f"Output dir: {output_dir}")
    print()

    for entry in entries:
        label = entry.get("label", "unknown")
        steps_file = entry.get("steps_file")

        if not steps_file:
            print(f"  SKIP: {label} - no steps_file")
            fail_count += 1
            continue

        input_path = base_dir / steps_file
        if not input_path.exists():
            print(f"  SKIP: {label} - file not found: {input_path}")
            fail_count += 1
            continue

        output_path = output_dir / f"{label}.json"

        metadata = {
            "source": str(steps_file),
            "original_label": label,
        }
        if entry.get("region_metadata_file"):
            metadata["region_metadata"] = entry["region_metadata_file"]
        if entry.get("sequencing_config"):
            metadata["sequencing_config"] = entry["sequencing_config"]

        if convert_single_file(input_path, output_path, label, metadata):
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert trajectory files to new JSON format"
    )

    # JSONL mode
    parser.add_argument(
        "--jsonl",
        type=Path,
        help="Path to JSONL config file listing trajectories (default: data/raw/dataset.jsonl)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Base directory for resolving relative paths in JSONL (default: data/raw)",
    )

    # Single file mode
    parser.add_argument(
        "--input",
        type=Path,
        help="Input txt file (single file mode)",
    )
    parser.add_argument(
        "--label",
        help="Label for the trajectory (single file mode)",
    )

    # Common
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file (single) or directory (jsonl mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/trajectories"),
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to convert",
    )

    args = parser.parse_args()

    # Determine script location to find data/raw
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if args.input:
        # Single file mode
        if not args.output:
            parser.error("--output is required for single file mode")

        label = args.label or args.input.stem
        success = convert_single_file(args.input, args.output, label)
        return 0 if success else 1

    else:
        # JSONL mode (default or explicit)
        jsonl_path = args.jsonl
        base_dir = args.base_dir

        # Default to data/raw/dataset.jsonl if no jsonl specified
        if jsonl_path is None:
            default_jsonl = project_root / "data" / "raw" / "dataset.jsonl"
            if default_jsonl.exists():
                jsonl_path = default_jsonl
                base_dir = base_dir or (project_root / "data" / "raw")
                print(f"Using default dataset: {jsonl_path}")
            else:
                parser.error(
                    f"No --jsonl specified and default not found at {default_jsonl}\n"
                    "Either provide --jsonl or ensure data/raw/dataset.jsonl exists"
                )

        base_dir = base_dir or Path(".")
        output_dir = args.output or args.output_dir

        success, failed = convert_from_jsonl(
            jsonl_path,
            base_dir,
            output_dir,
            args.max_files,
        )
        print()
        print(f"Done: {success} converted, {failed} failed")
        return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
