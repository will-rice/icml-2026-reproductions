"""Released-proof verification for Numina-Lean-Agent."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any


UPSTREAM_REVISION = (
    "github:project-numina/numina-lean-agent@"
    "1c9af8a52e715f22fede766425ba3d3b95526132+"
    "project-numina/Numina-Putnam2025@"
    "60d33c8ba19af905bd731e938ebde1c5b8c76519+"
    "project-numina/BrascampLieb@"
    "413f2bfd31100187eb6c2d632c9cbf12e3115494"
)
RELEASED_PROOF_SCOPE = "released-proof verification; not agent re-execution"
SORRY_PATTERN = re.compile(r"(?<![\w'])sorry(?![\w'])")


def invalidate_evidence(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)


def verify_clean_checkout(checkout: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if status:
        raise RuntimeError(f"checkout is not clean: {status.strip()}")


def tracked_lean_sources(checkout: Path) -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.lean"],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return sorted(Path(item) for item in output.split("\0") if item)


def _lean_code_without_comments_and_strings(source: str) -> str:
    code: list[str] = []
    index = 0
    block_depth = 0
    in_line_comment = False
    in_string = False
    escaped = False
    while index < len(source):
        pair = source[index : index + 2]
        character = source[index]
        if block_depth:
            if pair == "/-":
                block_depth += 1
                index += 2
            elif pair == "-/":
                block_depth -= 1
                index += 2
            else:
                if character == "\n":
                    code.append("\n")
                index += 1
        elif in_line_comment:
            if character == "\n":
                in_line_comment = False
                code.append("\n")
            index += 1
        elif in_string:
            if character == "\n":
                code.append("\n")
            if character == '"' and not escaped:
                in_string = False
            escaped = character == "\\" and not escaped
            if character != "\\":
                escaped = False
            index += 1
        elif pair == "/-":
            block_depth = 1
            index += 2
        elif pair == "--":
            in_line_comment = True
            index += 2
        elif character == '"':
            in_string = True
            escaped = False
            index += 1
        else:
            code.append(character)
            index += 1
    return "".join(code)


def scan_lean_sources(checkout: Path, relative_paths: list[Path]) -> dict[str, Any]:
    files_with_sorry: dict[str, int] = {}
    for relative_path in sorted(relative_paths):
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"invalid Lean source path: {relative_path}")
        code = _lean_code_without_comments_and_strings(
            (checkout / relative_path).read_text()
        )
        count = len(SORRY_PATTERN.findall(code))
        if count:
            files_with_sorry[relative_path.as_posix()] = count
    return {
        "file_count": len(relative_paths),
        "files_with_sorry": files_with_sorry,
        "method": "nested-comment/string-aware sorry token scan",
        "sorry_count": sum(files_with_sorry.values()),
    }
