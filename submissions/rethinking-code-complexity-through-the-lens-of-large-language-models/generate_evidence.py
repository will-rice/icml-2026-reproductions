from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lmcc_repro.evidence import write_evidence_bundle  # noqa: E402


if __name__ == "__main__":
    write_evidence_bundle(ROOT / "evidence" / "bundle.json")
