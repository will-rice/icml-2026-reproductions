from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tedbench_repro.evidence import build_bundle


def main() -> None:
    evidence_dir = PROJECT / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    (evidence_dir / "bundle.json").write_text(
        json.dumps(build_bundle(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
