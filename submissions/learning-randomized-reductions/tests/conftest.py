import os
from pathlib import Path
import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def cache_dir(project_root: Path) -> Path:
    uv_cache = os.environ.get("UV_CACHE_DIR")
    if uv_cache:
        uv_lrr_cache = Path(uv_cache) / "lrr-upstream"
        if uv_lrr_cache.exists():
            return uv_lrr_cache
    return project_root / ".cache/upstream"
