from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top_w_repro.evidence import build_bundle
from top_w_repro.pages import build_pages


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bundle = build_bundle()

    evidence_dir = ROOT / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    bundle_path = evidence_dir / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    logging.info("Evidence bundle written to %s", bundle_path)

    pages_dir = ROOT / "pages"
    pages_dir.mkdir(exist_ok=True)
    for name, content in build_pages(bundle).items():
        (pages_dir / name).write_text(content)
        logging.info("Page written to %s", pages_dir / name)

    for audit_name, audit in bundle["audits"].items():
        if not audit["passed"]:
            raise SystemExit(f"audit failed: {audit_name}")
    logging.info("All audits passed.")


if __name__ == "__main__":
    main()
