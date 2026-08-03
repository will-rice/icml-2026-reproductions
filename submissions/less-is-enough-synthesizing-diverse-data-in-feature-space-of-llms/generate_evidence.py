import json
from pathlib import Path

from fac_evidence.bundle import build_evidence_bundle
from fac_evidence.report import render_report


def main() -> None:
    root = Path(__file__).resolve().parent
    evidence_dir = root / "evidence"
    pages_dir = root / "pages"
    evidence_dir.mkdir(exist_ok=True)
    pages_dir.mkdir(exist_ok=True)
    bundle = build_evidence_bundle()
    (evidence_dir / "manifest.json").write_text(json.dumps(bundle["manifest"], indent=2, sort_keys=True) + "\n")
    (evidence_dir / "claims.json").write_text(json.dumps(bundle["claims"], indent=2, sort_keys=True) + "\n")
    (pages_dir / "report.md").write_text(render_report())


if __name__ == "__main__":
    main()
