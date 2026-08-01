"""Generate the deterministic ThreadWeaver evidence bundle."""

from pathlib import Path
import os

from threadweaver_repro.evidence import write_evidence


def default_upstream() -> Path:
    env_path = os.environ.get("THREADWEAVER_UPSTREAM")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[2] / "scratch" / "threadweaver-upstream"


def main() -> None:
    output = Path(__file__).resolve().parent / "evidence" / "bundle.json"
    evidence = write_evidence(default_upstream(), output)
    print(f"wrote {output}")
    print(f"paper_id={evidence['paper_id']} revision={evidence['upstream']['git_revision']}")


if __name__ == "__main__":
    main()
