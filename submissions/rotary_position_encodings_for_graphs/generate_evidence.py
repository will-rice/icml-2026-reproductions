#!/usr/bin/env python3
"""CLI script to generate evidence for Rotary Position Encodings for Graphs (WIRE)."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from submissions.rotary_position_encodings_for_graphs.wire import generate_evidence

def main():
    output_path = Path(__file__).resolve().parent / "evidence.json"
    print(f"Generating evidence to {output_path}...")
    data = generate_evidence(str(output_path))
    print("Evidence Generation Complete:")
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
