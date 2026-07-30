import json
from pathlib import Path
from llapdiffusion_repro.evidence import generate_bundle

def main():
    bundle = generate_bundle()
    output_path = Path(__file__).resolve().parent / "evidence" / "bundle.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(bundle, f, indent=2)
    print(output_path)

if __name__ == "__main__":
    main()
