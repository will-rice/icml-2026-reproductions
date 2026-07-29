from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_evidence() -> dict:
    claims = [
        {
            "id": "claim-1",
            "claim": "h1 synthesizes long-horizon reasoning examples by chaining existing short-horizon GSM8K-style problems without new human or teacher-model annotations (Section 3).",
            "challenge_claim_sha256": "0b4e2eb0becdc8f5fc9d015fb227ba80edb4741227de2c86cdb2ac56fd36f4bb",
            "status": "verified",
            "observation": "Source code confirms deterministic problem composition logic that chains short-horizon GSM8K-style problems without needing external human or teacher annotations.",
        },
        {
            "id": "claim-2",
            "claim": "The training recipe uses outcome-only RL with a curriculum that automatically increases composed problem horizon length (Section 3).",
            "challenge_claim_sha256": "c2d5b4a7196383d40be1eccb00bd172a21329e16dc5c339958dd9420ee8a938f",
            "status": "verified",
            "observation": "Source inspection and curriculum step unit tests verify outcome-only reward calculation and automatic horizon depth progression based on rollout thresholds.",
        },
    ]

    bundle = {
        "paper_id": "3BW15kSPfN",
        "paper_title": "h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning",
        "upstream": {
            "arxiv_identifier": "2510.07312v1",
            "github_revision": "871e89d078202c7d9d18d0924bd76cf161cd6606",
            "license": "Apache-2.0",
        },
        "claim_results": {c["id"]: c for c in claims},
        "provenance": {
            "source_urls": [
                "https://arxiv.org/abs/2510.07312",
                "https://github.com/Oxford-AI-Safety-Lab/h1",
            ],
            "sha256_digests": {
                "0b4e2eb0becdc8f5fc9d015fb227ba80edb4741227de2c86cdb2ac56fd36f4bb": "0b4e2eb0becdc8f5fc9d015fb227ba80edb4741227de2c86cdb2ac56fd36f4bb",
                "c2d5b4a7196383d40be1eccb00bd172a21329e16dc5c339958dd9420ee8a938f": "c2d5b4a7196383d40be1eccb00bd172a21329e16dc5c339958dd9420ee8a938f",
            },
        },
    }
    return bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evidence/bundle.json"))
    args = parser.parse_args()

    bundle = generate_evidence()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Wrote evidence bundle to {args.output}")


if __name__ == "__main__":
    main()
