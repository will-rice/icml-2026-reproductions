"""Hugging Face Space App for Rotary Position Encodings for Graphs (WIRE) reproduction."""

import json
from pathlib import Path
from submissions.rotary_position_encodings_for_graphs.wire import generate_evidence

def main():
    evidence_path = Path(__file__).resolve().parent / "evidence.json"
    data = generate_evidence(str(evidence_path))
    print("WIRE Space App Initialized.")
    print("Verified Claims:")
    for claim in data.get("claims", []):
        print(f"  - [{claim['status'].upper()}] {claim['claim_id']}: {claim['observation']}")

if __name__ == "__main__":
    main()
