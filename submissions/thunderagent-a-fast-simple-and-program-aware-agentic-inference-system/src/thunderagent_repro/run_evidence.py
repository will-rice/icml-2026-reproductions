"""Generate the ThunderAgent evidence bundle."""

from __future__ import annotations

from pathlib import Path

from .source_audit import write_evidence


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    source_root = project_root / "fixtures" / "ThunderAgent"
    output_path = project_root / "evidence" / "results.json"
    write_evidence(source_root, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
