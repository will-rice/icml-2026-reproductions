"""Pytest fixtures for FlexRank reproduction tests."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent
