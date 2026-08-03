"""Bounded, network-isolated decoding of pinned passive-video inputs."""

import base64
import hashlib
import io
import json
import os
import resource
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, Sequence

import numpy as np

MAX_INPUT_BYTES = 4 * 1024**3
MAX_FRAMES = 100_000
MAX_DIMENSION = 4096
WALL_SECONDS = 600
MAX_ADDRESS_SPACE = 8 * 1024**3


class MediaUnavailable(ValueError):
    """A video could not be safely and completely decoded."""


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    height: int
    width: int
    sha256: str
    size_bytes: int


def probe_video(path: Path, *, expected_sha256: str) -> VideoInfo:
    """Validate content identity and return bounded decoder metadata."""
    identity = _validate_input(path, expected_sha256)
    result = _run_child(path, expected_sha256, [])
    return VideoInfo(
        frame_count=_bounded_integer(result, "frame_count", 1, MAX_FRAMES),
        height=_bounded_integer(result, "height", 1, MAX_DIMENSION),
        width=_bounded_integer(result, "width", 1, MAX_DIMENSION),
        sha256=identity[0],
        size_bytes=identity[1],
    )


def decode_anchor_frames(
    path: Path,
    indices: Sequence[int],
    *,
    expected_sha256: str,
) -> np.ndarray:
    """Decode exactly one strictly increasing set of RGB frame indices."""
    _validate_input(path, expected_sha256)
    requested = list(indices)
    if (
        not requested
        or any(type(index) is not int or index < 0 for index in requested)
        or requested != sorted(set(requested))
    ):
        raise MediaUnavailable("indices must be unique, sorted, and nonnegative")
    result = _run_child(path, expected_sha256, requested)
    try:
        payload = base64.b64decode(result["frames_npy"], validate=True)
        frames = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as error:
        raise MediaUnavailable("invalid complete frame payload") from error
    expected_shape = (
        len(requested),
        _bounded_integer(result, "height", 1, MAX_DIMENSION),
        _bounded_integer(result, "width", 1, MAX_DIMENSION),
        3,
    )
    if frames.dtype != np.uint8 or frames.shape != expected_shape:
        raise MediaUnavailable("decoded frames violate RGB schema")
    return frames


def _validate_input(path: Path, expected_sha256: str) -> tuple[str, int]:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise MediaUnavailable("invalid expected video hash")
    if not path.is_file() or path.is_symlink():
        raise MediaUnavailable("video must be a regular file")
    size = path.stat().st_size
    if size < 1 or size > MAX_INPUT_BYTES:
        raise MediaUnavailable("video exceeds input size limit")
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise MediaUnavailable("video hash mismatch")
    return observed, size


def _run_child(
    path: Path, expected_sha256: str, indices: list[int]
) -> dict[str, object]:
    source_root = Path(__file__).resolve().parents[1]
    runtime = Path(sys.prefix).resolve()
    python_root = Path(sys.base_prefix).resolve()
    command = [
        "timeout",
        "--signal=KILL",
        str(WALL_SECONDS),
        "bwrap",
        "--die-with-parent",
        "--unshare-all",
        "--new-session",
        "--ro-bind",
        str(path.resolve()),
        "/input/video.mp4",
        "--ro-bind",
        str(source_root),
        "/package",
        "--ro-bind",
        str(runtime),
        "/runtime",
        "--ro-bind",
        str(python_root),
        "/python",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/usr/lib",
        "/usr/lib",
        "--tmpfs",
        "/tmp",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--setenv",
        "PYTHONPATH",
        "/package:/runtime/lib/python3.12/site-packages",
        "--setenv",
        "PYTHONHOME",
        "/python",
        "--setenv",
        "CUDA_VISIBLE_DEVICES",
        "",
        "--setenv",
        "OMP_NUM_THREADS",
        "2",
        "/python/bin/python3.12",
        "-m",
        "timerewarder_repro.media",
        "child",
        expected_sha256,
        json.dumps(indices, separators=(",", ":")),
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=WALL_SECONDS + 10,
            check=False,
            close_fds=True,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MediaUnavailable(f"media subprocess failed: {type(error).__name__}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        category = detail[-1][:200] if detail else f"exit-{completed.returncode}"
        raise MediaUnavailable(f"media subprocess returned unavailable: {category}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise MediaUnavailable("media subprocess returned invalid output") from error
    if not isinstance(result, dict):
        raise MediaUnavailable("media subprocess returned invalid output")
    return result


def _child(expected_sha256: str, indices_json: str) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (MAX_ADDRESS_SPACE, MAX_ADDRESS_SPACE))
    resource.setrlimit(resource.RLIMIT_CPU, (WALL_SECONDS, WALL_SECONDS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    path = Path("/input/video.mp4")
    if _sha256_file(path) != expected_sha256:
        raise MediaUnavailable("child video hash mismatch")
    indices = json.loads(indices_json)
    if not isinstance(indices, list) or any(type(index) is not int for index in indices):
        raise MediaUnavailable("child indices")

    import decord

    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=2)
    frame_count = len(reader)
    if frame_count < 1 or frame_count > MAX_FRAMES:
        raise MediaUnavailable("frame count limit")
    first = reader[0].asnumpy()
    if (
        first.ndim != 3
        or first.shape[2] != 3
        or max(first.shape[:2]) > MAX_DIMENSION
    ):
        raise MediaUnavailable(f"video dimension limit: {tuple(first.shape)}")
    result: dict[str, object] = {
        "frame_count": frame_count,
        "height": int(first.shape[0]),
        "width": int(first.shape[1]),
    }
    if indices:
        if indices != sorted(set(indices)) or min(indices) < 0 or max(indices) >= frame_count:
            raise MediaUnavailable("frame index unavailable")
        frames = reader.get_batch(indices).asnumpy()
        buffer = io.BytesIO()
        np.save(buffer, frames, allow_pickle=False)
        result["frames_npy"] = base64.b64encode(buffer.getvalue()).decode("ascii")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


def _bounded_integer(
    value: dict[str, object], key: str, lower: int, upper: int
) -> int:
    observed = value.get(key)
    if type(observed) is not int or not lower <= observed <= upper:
        raise MediaUnavailable(f"invalid {key}")
    return observed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _main() -> NoReturn:
    if len(sys.argv) != 4 or sys.argv[1] != "child":
        raise SystemExit("media child arguments")
    _child(sys.argv[2], sys.argv[3])
    raise SystemExit(0)


if __name__ == "__main__":
    _main()
