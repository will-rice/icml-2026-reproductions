import json
from pathlib import Path

from dbfm_repro.evidence import generate_evidence_bundle


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "evidence" / "bundle.json"


def _load_bundle() -> dict:
    if not BUNDLE.exists():
        generate_evidence_bundle(ROOT / "evidence")
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    bundle = _load_bundle()
    print(json.dumps(bundle, indent=2, sort_keys=True))
