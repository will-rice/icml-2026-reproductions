from pathlib import Path

from agent_primitives_repro.evidence import write_evidence


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    bundle_path, report_path = write_evidence(root)
    print(f"wrote {bundle_path}")
    print(f"wrote {report_path}")
