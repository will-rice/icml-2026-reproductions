import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'src'
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SRC_PARENT = Path(__file__).resolve().parent.parent / 'src'
if SRC_PARENT.exists() and str(SRC_PARENT) not in sys.path:
    sys.path.insert(0, str(SRC_PARENT))
