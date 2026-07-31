from __future__ import annotations

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sleeplm_repro.evidence import build_bundle


def main() -> None:
    out_dir = ROOT / "evidence"
    out_dir.mkdir(exist_ok=True)
    bundle = build_bundle()
    (out_dir / "bundle.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
