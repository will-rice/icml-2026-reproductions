"""Generate evidence bundle JSON for CapBencher reproduction."""

import json
from pathlib import Path
from capbencher.simulation import generate_evidence_bundle

if __name__ == "__main__":
    bundle = generate_evidence_bundle()
    evidence_dir = Path(__file__).parent / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    out_file = evidence_dir / "bundle.json"
    with open(out_file, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"Evidence bundle written to {out_file}")
