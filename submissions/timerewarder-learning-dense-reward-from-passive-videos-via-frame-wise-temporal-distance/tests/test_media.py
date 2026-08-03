import hashlib
import inspect
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from timerewarder_repro.media import (
    MediaUnavailable,
    decode_anchor_frames,
    probe_video,
)


def _video(tmp_path: Path) -> tuple[Path, str]:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the bounded media integration test")
    path = tmp_path / "fixture.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=1:duration=5",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_probe_and_decode_are_deterministic_rgb(tmp_path: Path) -> None:
    path, digest = _video(tmp_path)

    info = probe_video(path, expected_sha256=digest)
    first = decode_anchor_frames(path, [0, 1, 2, 3, 4], expected_sha256=digest)
    second = decode_anchor_frames(path, [0, 1, 2, 3, 4], expected_sha256=digest)

    assert info.frame_count == 5
    assert (info.height, info.width) == (48, 64)
    assert first.shape == (5, 48, 64, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)


def test_hash_and_index_failures_return_typed_unavailable(tmp_path: Path) -> None:
    path, digest = _video(tmp_path)

    with pytest.raises(MediaUnavailable, match="hash"):
        probe_video(path, expected_sha256="0" * 64)
    for indices in ([0, 0], [1, 0], [0, 5]):
        with pytest.raises(MediaUnavailable):
            decode_anchor_frames(path, indices, expected_sha256=digest)


def test_media_source_declares_resource_and_network_boundaries() -> None:
    import timerewarder_repro.media as media_module

    source = inspect.getsource(media_module)
    for boundary in (
        "4 * 1024**3",
        "100_000",
        "4096",
        "600",
        "8 * 1024**3",
        "--unshare-all",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
    ):
        assert boundary in source
