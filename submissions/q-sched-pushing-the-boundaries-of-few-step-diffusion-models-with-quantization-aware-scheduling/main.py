import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qsched.eval import run_evaluation


def main():
    print("Running Q-Sched Reproduction Benchmark...")
    results = run_evaluation()
    print("Evaluation completed cleanly.")
    print(json.dumps(results, indent=2))

    evidence_path = Path(__file__).resolve().parent / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evidence written to {evidence_path}")


if __name__ == "__main__":
    main()
