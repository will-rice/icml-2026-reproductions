#!/usr/bin/env python3
"""Generate the 3ViewSense evidence bundle from a pinned upstream checkout."""

from __future__ import annotations

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from viewsense_repro.claims import write_evidence_bundle, write_report_pages  # noqa: E402


def default_upstream_root() -> Path:
    env_path = os.environ.get("VIEWSENSE_UPSTREAM_ROOT")
    if env_path:
        return Path(env_path)
    return WORKSPACE_ROOT / "scratch" / "3viewsense-upstream"


def main() -> None:
    upstream = default_upstream_root()
    output = PROJECT_ROOT / "evidence.json"
    bundle = write_evidence_bundle(upstream, output)
    write_report_pages(bundle, PROJECT_ROOT / "pages")
    print(output)


if __name__ == "__main__":
    main()
