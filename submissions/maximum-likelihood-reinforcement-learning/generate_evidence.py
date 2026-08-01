from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from maxrl_repro import build_evidence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    evidence = build_evidence(args.source_root, args.output_dir)
    print({"claims": len(evidence["claims"]), "output_dir": str(args.output_dir)})


if __name__ == "__main__":
    main()
