from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from neural_thickets_repro.evidence import write_evidence



def _status_table(bundle: dict) -> str:
    rows = ["| Claim | Local status | Evidence note |", "| --- | --- | --- |"]
    for result in bundle["claim_results"]:
        rows.append(
            f"| {result['claim_index']} | `{result['status']}` | {result['evidence']} |"
        )
    return "\n".join(rows)


def write_report(bundle: dict) -> None:
    pages = ROOT / "pages"
    pages.mkdir(exist_ok=True)
    report = pages / "report.md"
    report.write_text(
        "\n".join(
            [
                "# Neural Thickets Reproduction Logbook",
                "",
                f"Attempt: `{bundle['attempt_id']}`",
                f"Paper: `{bundle['paper_id']}`",
                f"Snapshot: `{bundle['snapshot_id']}`",
                f"Upstream commit: `{bundle['upstream_commit_observed']}`",
                "",
                "## Claim Outcomes",
                "",
                _status_table(bundle),
                "",
                "## Deterministic Toy Observation",
                "",
                f"- Small-model proxy density: `{bundle['simulation']['small_model_density']}`",
                f"- Large-model proxy density: `{bundle['simulation']['large_model_density']}`",
                f"- Ensemble accuracy proxy: `{bundle['simulation']['ensemble_accuracy']}`",
                f"- Best single perturbation proxy: `{bundle['simulation']['best_single_accuracy']}`",
                "",
                "## Limitations",
                "",
                "\n".join(f"- {item}" for item in bundle["limitations"]),
                "",
                "Machine-readable evidence is in `evidence/bundle.json`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Neural Thickets reproduction evidence.")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence" / "bundle.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = write_evidence(args.output)
    write_report(bundle)
    print(f"wrote {args.output}")
    print(f"claims={len(bundle['claim_results'])}")


if __name__ == "__main__":
    main()
