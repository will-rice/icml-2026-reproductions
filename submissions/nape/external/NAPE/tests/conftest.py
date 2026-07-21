"""Shared test fixtures for behavioral equivalence tests."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


# Base paths
NEXT_ACTION_DIR = Path(__file__).parent.parent
DATA_DIR = NEXT_ACTION_DIR / "data"
TRAJECTORIES_DIR = DATA_DIR / "trajectories"


@pytest.fixture
def empty_state() -> Dict[str, Any]:
    """Empty workbook state."""
    return {"worksheets": {}}


@pytest.fixture
def trajectory_0000afae() -> List[str]:
    """Load the 0000afae trajectory operations."""
    path = TRAJECTORIES_DIR / "0000afae.json"
    if not path.exists():
        pytest.skip(f"Trajectory file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["operations"]
