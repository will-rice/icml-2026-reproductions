import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from par_protein.evidence import generate_evidence

if __name__ == "__main__":
    evidence_dir = Path(__file__).resolve().parent / "evidence"
    results_path = generate_evidence(evidence_dir)
    print(f"Generated evidence at: {results_path}")
