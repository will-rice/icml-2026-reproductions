import json
from pathlib import Path


def load_results() -> dict:
    path = Path(__file__).parent / "evidence" / "results.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"error": "Run python -m thunderagent_repro.run_evidence to generate evidence/results.json"}


if __name__ == "__main__":
    print(json.dumps(load_results(), indent=2, sort_keys=True))
