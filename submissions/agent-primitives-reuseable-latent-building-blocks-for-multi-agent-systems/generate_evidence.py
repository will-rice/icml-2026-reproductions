import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_primitives_repro.evidence import write_evidence


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    bundle_path, report_path = write_evidence(root)
    print(f"wrote {bundle_path}")
    print(f"wrote {report_path}")
