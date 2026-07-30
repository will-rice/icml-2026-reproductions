from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from scale_repro.evidence import write_evidence_bundle, write_provenance



def main():
    write_evidence_bundle(ROOT / "evidence" / "results.json")
    write_provenance(ROOT / "evidence" / "provenance.json")


if __name__ == "__main__":
    main()
