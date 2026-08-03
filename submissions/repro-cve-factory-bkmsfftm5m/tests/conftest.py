import sys
from pathlib import Path


submission_dir = Path(__file__).parent.parent
if str(submission_dir) not in sys.path:
    sys.path.insert(0, str(submission_dir))
