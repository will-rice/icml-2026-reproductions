import argparse
from pathlib import Path
from reward_free_alignment.evidence import build_evidence, write_evidence_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RACO reproduction evidence bundle")
    parser.add_argument("--output", type=Path, default=None, help="Output path for evidence JSON")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    evidence_data = build_evidence(project_root)

    output_path = args.output
    if output_path is None:
        output_path = project_root / "evidence/results.json"

    write_evidence_atomic(output_path, evidence_data)
    print(f"Evidence written to {output_path}")


if __name__ == "__main__":
    main()
