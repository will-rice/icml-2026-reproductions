from __future__ import annotations

import argparse
import json

from d3llm_repro.evidence import generate_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="evidence")
    args = parser.parse_args()
    bundle = generate_bundle(args.output_dir)
    print(json.dumps({"bundle": f"{args.output_dir}/bundle.json", "claims": len(bundle["claims"])}, sort_keys=True))


if __name__ == "__main__":
    main()
