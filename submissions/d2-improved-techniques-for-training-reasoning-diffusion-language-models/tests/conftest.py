import sys
from pathlib import Path

# Ensure src/ is in sys.path when pytest runs from workspace root
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
