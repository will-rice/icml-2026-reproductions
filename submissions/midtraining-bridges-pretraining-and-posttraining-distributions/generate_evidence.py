"""Evidence generator script for Midtraining Bridges."""

import json
from pathlib import Path
from midtraining_bridges.core import run_full_reproduction


def main():
    evidence = run_full_reproduction()
    output_path = Path(__file__).parent / "evidence.json"
    output_path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Generated evidence at {output_path}")


if __name__ == "__main__":
    main()
