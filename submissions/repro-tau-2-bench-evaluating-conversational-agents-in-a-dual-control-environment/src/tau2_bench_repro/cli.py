from __future__ import annotations

import argparse
from pathlib import Path

from tau2_bench_repro.evidence import resolve_upstream_root, write_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=None,
        help="Path to the pinned tau2-bench checkout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence.json"),
        help="Output evidence JSON path.",
    )
    args = parser.parse_args()
    upstream_root = args.upstream_root or resolve_upstream_root(Path.cwd())
    evidence = write_evidence(upstream_root, args.output)
    statuses = {claim["claim_id"]: claim["status"] for claim in evidence["claims"]}
    print(statuses)


if __name__ == "__main__":
    main()
