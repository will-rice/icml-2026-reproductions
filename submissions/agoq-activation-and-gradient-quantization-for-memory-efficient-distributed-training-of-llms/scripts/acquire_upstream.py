#!/usr/bin/env python3
"""Acquire pinned upstream source files from the AGoQ repository.

Usage:
    python scripts/acquire_upstream.py \
        --repository-url https://github.com/Eutenacity/AGoQ.git \
        --revision 006fa0f6318228d1fcd6727f0578c0e548e5cbff \
        --manifest evidence/inputs/upstream_manifest.json \
        --output evidence/inputs/upstream
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def git_blob_id(payload: bytes) -> str:
    """Compute a Git blob object ID from raw file bytes."""
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def canonical_sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire and verify pinned upstream source files."
    )
    parser.add_argument("--repository-url", required=True, help="Git repository URL")
    parser.add_argument("--revision", required=True, help="Exact commit SHA")
    parser.add_argument(
        "--manifest", required=True, help="Path to upstream manifest JSON"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for source files"
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Validate manifest matches arguments
    if manifest["repository"] != args.repository_url:
        print(
            f"ERROR: manifest repository {manifest['repository']!r} "
            f"does not match --repository-url {args.repository_url!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    if manifest["commit"] != args.revision:
        print(
            f"ERROR: manifest commit {manifest['commit']!r} "
            f"does not match --revision {args.revision!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Initialize a bare repo and fetch the exact revision
        subprocess.run(
            ["git", "init", "--bare", str(tmp / "repo.git")],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp / "repo.git"),
                "fetch",
                "--depth=1",
                args.repository_url,
                args.revision,
            ],
            check=True,
            capture_output=True,
        )

        errors = []
        for entry in manifest["files"]:
            path = entry["path"]
            expected_sha256 = entry["sha256"]
            expected_blob = entry["git_blob"]
            expected_size = entry["size_bytes"]

            # Extract file content
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(tmp / "repo.git"),
                    "show",
                    f"FETCH_HEAD:{path}",
                ],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                errors.append(f"Failed to extract {path}: {result.stderr.decode()}")
                continue

            content = result.stdout

            # Verify size
            if len(content) != expected_size:
                errors.append(
                    f"{path}: size {len(content)} != expected {expected_size}"
                )
                continue

            # Verify SHA-256
            actual_sha256 = canonical_sha256(content)
            if actual_sha256 != expected_sha256:
                errors.append(
                    f"{path}: SHA-256 {actual_sha256} != expected {expected_sha256}"
                )
                continue

            # Verify Git blob ID
            actual_blob = git_blob_id(content)
            if actual_blob != expected_blob:
                errors.append(
                    f"{path}: git blob {actual_blob} != expected {expected_blob}"
                )
                continue

            # Write to a temporary file first, then atomically move
            out_path = output_dir / path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_out = out_path.with_suffix(".tmp")
            tmp_out.write_bytes(content)
            tmp_out.replace(out_path)
            print(f"  OK {path} ({len(content)} bytes)")

        if errors:
            print("\nERRORS:", file=sys.stderr)
            for err in errors:
                print(f"  {err}", file=sys.stderr)
            sys.exit(1)

    print(f"\nAll {len(manifest['files'])} files verified and written to {output_dir}")


if __name__ == "__main__":
    main()
