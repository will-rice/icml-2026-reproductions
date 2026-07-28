import pathlib
import pytest


@pytest.fixture
def project_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent
