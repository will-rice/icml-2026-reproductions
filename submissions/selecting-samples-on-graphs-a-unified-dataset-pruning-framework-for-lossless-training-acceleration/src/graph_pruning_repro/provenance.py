"""Pinned paper identity and authenticated transcription loading."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

PAPER = {
    "challenge_id": "a3GdvuPItd",
    "revision": "arxiv:2606.12913v2",
    "source_url": "https://arxiv.org/pdf/2606.12913v2",
    "pdf_byte_count": 683737,
    "pdf_sha256": (
        "26ce80e8d347340e0055f2bcf061b6b3e29489fc68a85b8d5711e12cc9da5090"
    ),
    "license": "CC BY-NC-SA 4.0",
}

TARGET_CLAIMS = (
    "The paper casts dataset pruning as a graph problem with node weights "
    "for intrinsic importance and edge weights for extrinsic "
    "diversity/interaction, yielding a Maximum Weight Clique formulation "
    "(Section 3.3).",
    "Under mild conditions, the unified objective becomes submodular and "
    "admits a greedy approximation guarantee (Section 3.6; Appendix F).",
)

PDF_ACQUISITION_COMMAND = (
    "curl --fail --location --proto '=https' --tlsv1.2 "
    "--output /tmp/2606.12913v2.pdf "
    "https://export.arxiv.org/pdf/2606.12913v2 && "
    'test "$(wc -c < /tmp/2606.12913v2.pdf)" -eq 683737 && '
    "printf '%s  %s\\n' "
    "26ce80e8d347340e0055f2bcf061b6b3e29489fc68a85b8d5711e12cc9da5090 "
    "/tmp/2606.12913v2.pdf | sha256sum --check --strict"
)

TRANSCRIPTION_SET_SHA256 = (
    "b2bb4563ecb883ef1cbacbd80575c8e2467eaa0331b6b2c9283a39e1d04e1454"
)

_MANIFEST_KEYS = {
    "record_id",
    "equation",
    "pdf_page",
    "section",
    "normalized_expression",
    "source_excerpt_path",
    "source_excerpt_byte_count",
    "source_excerpt_sha256",
    "reviewed_by",
}
_APPROVED_REVIEWERS = (
    "codex-graph-pruning-design-author-v2",
    "codex-graph-pruning-design-reviewer-v2",
)
_APPROVED_RECORD_METADATA = (
    (
        "eq-02",
        "2",
        2,
        "3.3 Rethinking Pruning from a Graph Perspective",
        "w_i = alpha I_in(x_i); a_ij = g(D(x_i,x_j))",
        "paper_transcriptions/excerpts/eq-02.txt",
    ),
    (
        "eq-03",
        "3",
        3,
        "3.3 Maximum Weight Clique Formulation",
        "maximize sum_{i in C} w_i + sum_{{i,j} subseteq C} a_ij "
        "subject to |C|=b",
        "paper_transcriptions/excerpts/eq-03.txt",
    ),
    (
        "eq-04",
        "4",
        3,
        "3.3 Sample-wise Reformulation",
        "f(S) = sum_{x_i in S} [alpha I_in(x_i) + I_ex(x_i|S)] "
        "subject to |S|=b",
        "paper_transcriptions/excerpts/eq-04.txt",
    ),
    (
        "eq-05",
        "5",
        3,
        "3.3 Sample-wise Reformulation",
        "I_ex(x_i|S) = sum_{x_j in S minus {x_i}} a_ij = "
        "sum_{x_j in S minus {x_i}} g(D(x_i,x_j))",
        "paper_transcriptions/excerpts/eq-05.txt",
    ),
    (
        "eq-06",
        "6",
        3,
        "3.4 Greedy Selection with Unified Importance",
        "Delta_minus(v_i|G) = w_i + sum_{v_j in C minus {v_i}} a_ij",
        "paper_transcriptions/excerpts/eq-06.txt",
    ),
    (
        "eq-07",
        "7",
        3,
        "3.4 Unified Importance",
        "I(x_i|S) = Delta(x_i|S) = alpha I_in(x_i) + I_ex(x_i|S)",
        "paper_transcriptions/excerpts/eq-07.txt",
    ),
    (
        "eq-08",
        "8",
        3,
        "3.4 Greedy Selection Strategy",
        "x_star in argmax_{x_i in T minus S_t} I(x_i|S_t); "
        "S_{t+1} = S_t union {x_star}",
        "paper_transcriptions/excerpts/eq-08.txt",
    ),
    (
        "eq-10-11",
        "10-11",
        4,
        "3.6 Definition 3.3",
        "Delta(x|A) >= Delta(x|B), where Delta(x|A) = "
        "f(A union {x}) - f(A)",
        "paper_transcriptions/excerpts/eq-10-11.txt",
    ),
    (
        "eq-12-14",
        "12-14",
        4,
        "3.6 Lemma 3.4 proof",
        "Delta_A = alpha I_in + sum_A g; Delta_B = alpha I_in + sum_B g; "
        "Delta_A - Delta_B = -sum_{B minus A} g >= 0",
        "paper_transcriptions/excerpts/eq-12-14.txt",
    ),
    (
        "algorithm-1",
        "Algorithm 1",
        5,
        "Algorithm 1",
        "literal source lines 1-17 with PDF line wrapping normalized "
        "and no operational repair",
        "paper_transcriptions/algorithm1.txt",
    ),
    (
        "appendix-e-inline",
        "Appendix E inline",
        15,
        "Appendix E Maintain Monotonicity",
        "I_in_revised(x_i) = I_in(x_i) + sum_{j=1}^{|S_hat|} eta",
        "paper_transcriptions/excerpts/appendix-e-inline.txt",
    ),
    (
        "appendix-e-eq-26",
        "26",
        15,
        "Appendix E Maintain Monotonicity",
        "Delta(x_i|S_hat) = alpha I_in(x_i) + "
        "alpha sum_{j=1}^{|S_hat|} eta + "
        "sum_{x_j in S_hat} g(D(x_i,x_j))",
        "paper_transcriptions/excerpts/appendix-e-eq-26.txt",
    ),
    (
        "appendix-inline-literal-marginal",
        "Appendix E literal-derived marginal",
        15,
        "Approved design derivation from Appendix E and Eqs. 4-5",
        "Delta_appendix(x|S) = alpha I_in(x) + 2 sum_{j in S} a_xj + "
        "alpha eta (2|S|+1)",
        "paper_transcriptions/excerpts/appendix-inline-literal-marginal.txt",
    ),
    (
        "appendix-inline-single-marginal",
        "Appendix E single-counted-derived marginal",
        15,
        "Approved design derivation for the repaired single-counted objective",
        "Delta_single(x|S) = alpha I_in(x) + sum_{j in S} a_xj + "
        "alpha eta (2|S|+1)",
        "paper_transcriptions/excerpts/appendix-inline-single-marginal.txt",
    ),
    (
        "appendix-e-eq-27",
        "27",
        15,
        "Appendix E Maintain Monotonicity",
        "eta >= (1/alpha) max_{x_i,x_j} |g(D(x_i,x_j))|",
        "paper_transcriptions/excerpts/appendix-e-eq-27.txt",
    ),
    (
        "appendix-f-eq-28-38",
        "28-38",
        "16-17",
        "Appendix F Proof of the Greedy Approximation Guarantee",
        "literal Eq. 28-38 chain ending f(S_greedy) >= "
        "(1 - 1/e) f(S_star), without repairing its b-t or product steps",
        "paper_transcriptions/excerpts/appendix-f-eq-28-38.txt",
    ),
)


def verify_pdf(path: Path) -> None:
    """Reject any bytes that are not the pinned arXiv v2 PDF."""

    data = path.read_bytes()
    if len(data) != PAPER["pdf_byte_count"]:
        raise ValueError("pinned PDF byte count mismatch")
    if hashlib.sha256(data).hexdigest() != PAPER["pdf_sha256"]:
        raise ValueError("pinned PDF SHA-256 mismatch")


def transcription_set_sha256(
    records: Sequence[Mapping[str, object]],
) -> str:
    """Hash every field of the ordered transcription records."""

    payload = json.dumps(
        list(records),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"{label} is missing or unreadable") from exc
    if stat.S_ISLNK(mode):
        raise ValueError(f"{label} must not be a symlink")
    if not stat.S_ISDIR(mode):
        raise ValueError(f"{label} must be a directory")


def _require_regular_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"{label} is missing or unreadable") from exc
    if stat.S_ISLNK(mode):
        raise ValueError(f"{label} must not be a symlink")
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular file")


def _collect_logical_files(
    project_root: Path,
    transcription_root: Path,
) -> set[str]:
    pending = [transcription_root]
    logical_files: set[str] = set()

    while pending:
        directory = pending.pop()
        _require_directory(directory, "transcription directory")
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ValueError("transcription directory is unreadable") from exc

        for entry in entries:
            path = directory / entry.name
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise ValueError("transcription entry is unreadable") from exc
            logical_path = path.relative_to(project_root).as_posix()
            if stat.S_ISLNK(mode):
                raise ValueError(
                    f"transcription entry must not be a symlink: {logical_path}"
                )
            if stat.S_ISDIR(mode):
                pending.append(path)
            elif stat.S_ISREG(mode):
                logical_files.add(logical_path)
            else:
                raise ValueError(
                    f"transcription entry must be regular: {logical_path}"
                )

    return logical_files


def _resolve_excerpt_path(root: Path, raw_path: object) -> Path:
    if type(raw_path) is not str:
        raise ValueError("transcription excerpt path must be a string")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != raw_path
        or not relative.parts
        or relative.parts[0] != "paper_transcriptions"
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("transcription excerpt path is not canonical")

    transcription_root = root / "paper_transcriptions"
    logical_path = root.joinpath(*relative.parts)
    if not logical_path.is_relative_to(transcription_root):
        raise ValueError("transcription excerpt path escapes its root")

    directory = transcription_root
    _require_directory(directory, "transcription root")
    for part in relative.parts[1:-1]:
        directory /= part
        _require_directory(directory, "transcription path directory")
    _require_regular_file(logical_path, "transcription excerpt")
    return logical_path


def _validate_record_types(record: Mapping[str, object]) -> None:
    for field in (
        "record_id",
        "equation",
        "section",
        "normalized_expression",
        "source_excerpt_path",
        "source_excerpt_sha256",
    ):
        value = record[field]
        if type(value) is not str or not value.strip():
            raise ValueError(f"{field} must be a nonempty string")

    pdf_page = record["pdf_page"]
    if not (
        (type(pdf_page) is int and pdf_page > 0)
        or (type(pdf_page) is str and bool(pdf_page.strip()))
    ):
        raise ValueError("pdf_page must be a positive integer or nonempty string")

    byte_count = record["source_excerpt_byte_count"]
    if type(byte_count) is not int or byte_count <= 0:
        raise ValueError("source_excerpt_byte_count must be a positive integer")

    digest = record["source_excerpt_sha256"]
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("source_excerpt_sha256 must be lowercase SHA-256")

    reviewed_by = record["reviewed_by"]
    if (
        type(reviewed_by) is not list
        or tuple(reviewed_by) != _APPROVED_REVIEWERS
    ):
        raise ValueError("transcription reviewers do not match approved identities")


def _record_metadata(record: Mapping[str, object]) -> tuple[object, ...]:
    return (
        record["record_id"],
        record["equation"],
        record["pdf_page"],
        record["section"],
        record["normalized_expression"],
        record["source_excerpt_path"],
    )


def load_transcriptions(root: Path) -> tuple[dict[str, object], ...]:
    """Load and authenticate the complete file-backed transcription set."""

    try:
        project_root = root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("project root is missing or unreadable") from exc
    _require_directory(project_root, "project root")
    transcription_root = project_root / "paper_transcriptions"
    _require_directory(transcription_root, "transcription root")
    logical_files = _collect_logical_files(project_root, transcription_root)

    manifest_path = transcription_root / "manifest.json"
    _require_regular_file(manifest_path, "transcription manifest")
    try:
        manifest_text = manifest_path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("transcription manifest must be UTF-8") from exc
    try:
        raw_records = json.loads(
            manifest_text,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("transcription manifest is invalid JSON") from exc
    if type(raw_records) is not list:
        raise ValueError("transcription manifest must be an array")
    if len(raw_records) != len(_APPROVED_RECORD_METADATA):
        raise ValueError("transcription manifest record count mismatch")

    record_ids: set[str] = set()
    referenced_paths: set[str] = set()
    records: list[dict[str, object]] = []

    for index, raw_record in enumerate(raw_records):
        if type(raw_record) is not dict or set(raw_record) != _MANIFEST_KEYS:
            raise ValueError("transcription manifest record keys mismatch")
        record = dict(raw_record)
        _validate_record_types(record)

        record_id = record["record_id"]
        if record_id in record_ids:
            raise ValueError("duplicate transcription record ID")
        record_ids.add(record_id)

        raw_excerpt_path = record["source_excerpt_path"]
        if raw_excerpt_path in referenced_paths:
            raise ValueError("duplicate transcription excerpt path")
        referenced_paths.add(raw_excerpt_path)

        if _record_metadata(record) != _APPROVED_RECORD_METADATA[index]:
            raise ValueError("transcription approved metadata mismatch")

        excerpt_path = _resolve_excerpt_path(
            project_root,
            raw_excerpt_path,
        )
        excerpt = excerpt_path.read_bytes()
        try:
            excerpt.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("transcription excerpt must be UTF-8") from exc
        if record["source_excerpt_byte_count"] != len(excerpt):
            raise ValueError("transcription excerpt byte count mismatch")
        if record["source_excerpt_sha256"] != hashlib.sha256(excerpt).hexdigest():
            raise ValueError("transcription excerpt SHA-256 mismatch")

        records.append(record)

    expected_files = {"paper_transcriptions/manifest.json", *referenced_paths}
    if logical_files != expected_files:
        raise ValueError("transcription referenced-file set mismatch")

    result = tuple(records)
    if transcription_set_sha256(result) != TRANSCRIPTION_SET_SHA256:
        raise ValueError("transcription aggregate SHA-256 mismatch")
    return result
