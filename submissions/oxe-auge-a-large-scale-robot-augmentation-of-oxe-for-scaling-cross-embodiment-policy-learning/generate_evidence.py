import json
from pathlib import Path
from oxe_auge_repro.pipeline import generate_evidence_bundle

if __name__ == "__main__":
    bundle = generate_evidence_bundle()
    out_dir = Path(__file__).parent / "evidence"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "bundle.json"
    out_file.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote evidence bundle to {out_file}")
