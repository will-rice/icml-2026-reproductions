#!/usr/bin/env python3
"""Acquire upstream artifacts declared in upstream_manifest.json."""

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.request

from lrr_repro.provenance import git_blob_id, read_manifest, validate_manifest


def acquire_all(project_root: Path, cache_dir: Path) -> None:
    manifest = read_manifest(project_root)
    validate_manifest(manifest)

    cache_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = project_root / "evidence/inputs"

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for art in manifest["artifacts"]:
            art_id = art["artifact_id"]
            url = art["url"]
            expected_sha = art["sha256"]
            expected_size = art["size_bytes"]
            expected_blob = art.get("git_blob")
            rel_path = art.get("relative_path")

            if rel_path is not None:
                final_path = inputs_dir / rel_path
            else:
                if "v1" in art_id:
                    final_path = cache_dir / "2412.18134v1.pdf"
                elif "v5" in art_id:
                    final_path = cache_dir / "2412.18134v5.pdf"
                else:
                    final_path = cache_dir / f"{art_id}.pdf"

            if final_path.exists():
                payload = final_path.read_bytes()
                if (
                    len(payload) == expected_size
                    and hashlib.sha256(payload).hexdigest() == expected_sha
                ):
                    print(f"Skipping acquisition for verified {art_id} at {final_path}")
                    continue

            print(f"Downloading {art_id} from {url}...")
            tmp_file = tmpdir / f"download_{art_id}"
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (ICML 2026 Reproduction Agent Audit)"
                },
            )
            with urllib.request.urlopen(req) as resp, open(tmp_file, "wb") as f:
                shutil.copyfileobj(resp, f)

            payload = tmp_file.read_bytes()
            if len(payload) != expected_size:
                raise ValueError(
                    f"Downloaded {art_id} size mismatch: expected {expected_size}, got {len(payload)}"
                )
            actual_sha = hashlib.sha256(payload).hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Downloaded {art_id} SHA mismatch: expected {expected_sha}, got {actual_sha}"
                )
            if expected_blob is not None:
                actual_blob = git_blob_id(payload)
                if actual_blob != expected_blob:
                    raise ValueError(
                        f"Downloaded {art_id} Git blob mismatch: expected {expected_blob}, got {actual_blob}"
                    )

            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_file, final_path)
            print(f"Successfully installed {art_id} to {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire upstream inputs.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Submission project root directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / ".cache/upstream",
        help="PDF cache directory",
    )
    args = parser.parse_args()
    acquire_all(args.project_root, args.cache_dir)


if __name__ == "__main__":
    main()
