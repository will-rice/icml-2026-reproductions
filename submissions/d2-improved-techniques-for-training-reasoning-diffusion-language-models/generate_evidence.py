from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from d2_repro.evidence import main


if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--output" not in argv:
        argv = ["--output", str(Path(__file__).resolve().parent / "evidence" / "bundle.json")] + argv
    raise SystemExit(main(argv))
