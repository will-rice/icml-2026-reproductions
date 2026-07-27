import sys
from pathlib import Path

submission_root = Path(__file__).resolve().parent.parent
if str(submission_root) not in sys.path:
    sys.path.insert(0, str(submission_root))

src_dir = submission_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
