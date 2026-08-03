from __future__ import annotations

import argparse
from pathlib import Path

from skill_neologisms_repro.evidence import (
    build_evidence_bundle,
    render_summary_markdown,
    write_evidence_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Skill Neologisms reproduction evidence bundle."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--command", action="append", default=[])
    args = parser.parse_args()

    bundle = build_evidence_bundle(
        source_root=args.source_root,
        command_log=args.command,
    )
    bundle_path = write_evidence_bundle(bundle)
    summary_path = Path(__file__).resolve().parent / "pages" / "00-summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(render_summary_markdown(bundle), encoding="utf-8")
    print(f"Wrote {bundle_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
