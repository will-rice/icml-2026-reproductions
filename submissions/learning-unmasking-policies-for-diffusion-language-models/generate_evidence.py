import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from unmasking_policies_repro.evidence import write_evidence

if __name__ == "__main__":
    bundle_path, report_path = write_evidence(ROOT)
    print(f"wrote {bundle_path}")
    print(f"wrote {report_path}")
