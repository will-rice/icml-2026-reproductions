import json
from pathlib import Path


def load_bundle() -> dict:
    path = Path(__file__).parent / "evidence" / "bundle.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"error": "Run generate_evidence.py to create evidence/bundle.json"}


if __name__ == "__main__":
    print(json.dumps(load_bundle(), indent=2, sort_keys=True))