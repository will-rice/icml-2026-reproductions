#!/usr/bin/env python
"""
Launch the evaluation dashboard.

Usage:
    python scripts/run_dashboard.py --dir outputs/experiment_dir
    python scripts/run_dashboard.py --dir outputs/ --port 8501
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch evaluation dashboard")
    parser.add_argument(
        "--dir",
        type=str,
        default="results",
        help="Experiment directory or parent directory containing experiments",
    )
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port")
    args = parser.parse_args()

    app_path = (
        Path(__file__).parent.parent
        / "src"
        / "next_action_pred_eval"
        / "dashboard"
        / "app.py"
    )

    if not app_path.exists():
        print(f"Error: Dashboard app not found at {app_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--",
        "--dir",
        args.dir,
    ]

    print(f"Launching dashboard: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
