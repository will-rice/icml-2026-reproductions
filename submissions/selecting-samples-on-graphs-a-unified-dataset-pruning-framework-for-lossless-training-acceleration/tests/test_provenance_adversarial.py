import hashlib
import json
import shutil
from pathlib import Path

import pytest

from graph_pruning_repro.provenance import (
    TRANSCRIPTION_SET_SHA256,
    load_transcriptions,
    transcription_set_sha256,
    verify_pdf,
)


PROJECT_ROOT = Path(__file__).parents[1]
TRANSCRIPTION_DIGEST = (
    "b2bb4563ecb883ef1cbacbd80575c8e2467eaa0331b6b2c9283a39e1d04e1454"
)
REVIEWERS = (
    "codex-graph-pruning-design-author-v2",
    "codex-graph-pruning-design-reviewer-v2",
)
RECORD_FIELDS = (
    "record_id",
    "equation",
    "pdf_page",
    "section",
    "normalized_expression",
    "source_excerpt_path",
    "source_excerpt_byte_count",
    "source_excerpt_sha256",
)
APPROVED_RECORD_ROWS = (
    (
        "eq-02",
        "2",
        2,
        "3.3 Rethinking Pruning from a Graph Perspective",
        "w_i = alpha I_in(x_i); a_ij = g(D(x_i,x_j))",
        "paper_transcriptions/excerpts/eq-02.txt",
        49,
        "f9acf81ae1220aa0313c682459b868d0b44ae05f3374e39cba3400104bcf8cb5",
        REVIEWERS,
    ),
    (
        "eq-03",
        "3",
        3,
        "3.3 Maximum Weight Clique Formulation",
        "maximize sum_{i in C} w_i + sum_{{i,j} subseteq C} a_ij subject to |C|=b",
        "paper_transcriptions/excerpts/eq-03.txt",
        84,
        "b25985815cd92dc9831630d855abb1b6fd7c607497f811193d04fdc489cf7f65",
        REVIEWERS,
    ),
    (
        "eq-04",
        "4",
        3,
        "3.3 Sample-wise Reformulation",
        "f(S) = sum_{x_i in S} [alpha I_in(x_i) + I_ex(x_i|S)] subject to |S|=b",
        "paper_transcriptions/excerpts/eq-04.txt",
        88,
        "d9abff1676c7a1fd438f89b5e61ae5690bdff4434ea980671753da0201105fd7",
        REVIEWERS,
    ),
    (
        "eq-05",
        "5",
        3,
        "3.3 Sample-wise Reformulation",
        "I_ex(x_i|S) = sum_{x_j in S minus {x_i}} a_ij = "
        "sum_{x_j in S minus {x_i}} g(D(x_i,x_j))",
        "paper_transcriptions/excerpts/eq-05.txt",
        90,
        "73d5f5d03e9d2c4172043d716782506cbf11c6d08c8085495431f7cdf961a953",
        REVIEWERS,
    ),
    (
        "eq-06",
        "6",
        3,
        "3.4 Greedy Selection with Unified Importance",
        "Delta_minus(v_i|G) = w_i + sum_{v_j in C minus {v_i}} a_ij",
        "paper_transcriptions/excerpts/eq-06.txt",
        57,
        "7b5930c99bb2ca93bd548a39997da9542fec1c2a3a5eca9441f91453dbc2c2fc",
        REVIEWERS,
    ),
    (
        "eq-07",
        "7",
        3,
        "3.4 Unified Importance",
        "I(x_i|S) = Delta(x_i|S) = alpha I_in(x_i) + I_ex(x_i|S)",
        "paper_transcriptions/excerpts/eq-07.txt",
        62,
        "fc3af88dcc8950a9059f33c5039f59f8258c287fb2659c6036362bdfd9e14547",
        REVIEWERS,
    ),
    (
        "eq-08",
        "8",
        3,
        "3.4 Greedy Selection Strategy",
        "x_star in argmax_{x_i in T minus S_t} I(x_i|S_t); "
        "S_{t+1} = S_t union {x_star}",
        "paper_transcriptions/excerpts/eq-08.txt",
        83,
        "415d587f33dd8e64e19e6e0bd2ce21e354c825b25f4e9f2132bf39003c493d0c",
        REVIEWERS,
    ),
    (
        "eq-10-11",
        "10-11",
        4,
        "3.6 Definition 3.3",
        "Delta(x|A) >= Delta(x|B), where Delta(x|A) = "
        "f(A union {x}) - f(A)",
        "paper_transcriptions/excerpts/eq-10-11.txt",
        110,
        "94e946b6fd4d64f1b6361129ea6b2b2d51033d600e6e082257d6a4c21a116665",
        REVIEWERS,
    ),
    (
        "eq-12-14",
        "12-14",
        4,
        "3.6 Lemma 3.4 proof",
        "Delta_A = alpha I_in + sum_A g; Delta_B = alpha I_in + sum_B g; "
        "Delta_A - Delta_B = -sum_{B minus A} g >= 0",
        "paper_transcriptions/excerpts/eq-12-14.txt",
        229,
        "7e2fb8f8d99fd8dbbb93d49fbb312f790e24433bfc3e906e77df6c1615102f17",
        REVIEWERS,
    ),
    (
        "algorithm-1",
        "Algorithm 1",
        5,
        "Algorithm 1",
        "literal source lines 1-17 with PDF line wrapping normalized "
        "and no operational repair",
        "paper_transcriptions/algorithm1.txt",
        926,
        "d76e4ad27e1db3256341079e15911593851962da2e04bd8b5913cb774ee79249",
        REVIEWERS,
    ),
    (
        "appendix-e-inline",
        "Appendix E inline",
        15,
        "Appendix E Maintain Monotonicity",
        "I_in_revised(x_i) = I_in(x_i) + sum_{j=1}^{|S_hat|} eta",
        "paper_transcriptions/excerpts/appendix-e-inline.txt",
        52,
        "4efb0093f7436845bc7778566a0e12cecf78b625dad033daec92d5d61e68f7f4",
        REVIEWERS,
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
        92,
        "52401ab5513e102139e75816ec720e08de10d4b0f338bd7cccb84e4e7d13f85f",
        REVIEWERS,
    ),
    (
        "appendix-inline-literal-marginal",
        "Appendix E literal-derived marginal",
        15,
        "Approved design derivation from Appendix E and Eqs. 4-5",
        "Delta_appendix(x|S) = alpha I_in(x) + 2 sum_{j in S} a_xj + "
        "alpha eta (2|S|+1)",
        "paper_transcriptions/excerpts/appendix-inline-literal-marginal.txt",
        196,
        "e800eca6aa3a8b2e927e252b98ad918d54be689b5ccde0feafc1a6c193e2bfe9",
        REVIEWERS,
    ),
    (
        "appendix-inline-single-marginal",
        "Appendix E single-counted-derived marginal",
        15,
        "Approved design derivation for the repaired single-counted objective",
        "Delta_single(x|S) = alpha I_in(x) + sum_{j in S} a_xj + "
        "alpha eta (2|S|+1)",
        "paper_transcriptions/excerpts/appendix-inline-single-marginal.txt",
        145,
        "8bebae6d4a0a4d2c7d74be7f36ed913614915baf5a67b1db82841cdd8f340204",
        REVIEWERS,
    ),
    (
        "appendix-e-eq-27",
        "27",
        15,
        "Appendix E Maintain Monotonicity",
        "eta >= (1/alpha) max_{x_i,x_j} |g(D(x_i,x_j))|",
        "paper_transcriptions/excerpts/appendix-e-eq-27.txt",
        52,
        "0a3c6b4152ec34c49e90732c254d4400004cc4c728c0d84a1ba331098fa8bc04",
        REVIEWERS,
    ),
    (
        "appendix-f-eq-28-38",
        "28-38",
        "16-17",
        "Appendix F Proof of the Greedy Approximation Guarantee",
        "literal Eq. 28-38 chain ending f(S_greedy) >= "
        "(1 - 1/e) f(S_star), without repairing its b-t or product steps",
        "paper_transcriptions/excerpts/appendix-f-eq-28-38.txt",
        813,
        "a3aeac1dfb66659f37ec205f89862754633a3a84a7c3e993f1e96d3bc69b7be4",
        REVIEWERS,
    ),
)


def _copy_transcriptions(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    shutil.copytree(
        PROJECT_ROOT / "paper_transcriptions",
        project / "paper_transcriptions",
    )
    return project


def _manifest_path(project: Path) -> Path:
    return project / "paper_transcriptions" / "manifest.json"


def _read_manifest(project: Path) -> list[dict[str, object]]:
    return json.loads(_manifest_path(project).read_text(encoding="utf-8"))


def _write_manifest(
    project: Path,
    records: list[dict[str, object]],
) -> None:
    _manifest_path(project).write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )


def _approved_rows(
    records: tuple[dict[str, object], ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(record[field] for field in RECORD_FIELDS)
        + (tuple(record["reviewed_by"]),)
        for record in records
    )


def _replace_with_internal_symlink(path: Path) -> None:
    target = path.with_name(path.name + ".real")
    path.rename(target)
    path.symlink_to(target.name, target_is_directory=target.is_dir())


def test_pdf_rejects_same_size_tamper(tmp_path: Path) -> None:
    tampered = tmp_path / "same-size.pdf"
    tampered.write_bytes(b"\0" * 683737)

    with pytest.raises(ValueError, match="SHA-256"):
        verify_pdf(tampered)


def test_manifest_matches_literal_approved_record_mapping() -> None:
    records = load_transcriptions(PROJECT_ROOT)

    assert len(records) == 16
    assert _approved_rows(records) == APPROVED_RECORD_ROWS


def test_literal_full_record_aggregate_digest() -> None:
    records = json.loads(
        (PROJECT_ROOT / "paper_transcriptions" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert transcription_set_sha256(records) == TRANSCRIPTION_DIGEST
    assert TRANSCRIPTION_SET_SHA256 == TRANSCRIPTION_DIGEST


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("equation", "mutated equation"),
        ("pdf_page", 999),
        ("section", "mutated section"),
        ("normalized_expression", "mutated expression"),
    ),
)
def test_rejects_approved_metadata_mutation(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    project = _copy_transcriptions(tmp_path)
    records = _read_manifest(project)
    records[0][field] = value
    _write_manifest(project, records)

    with pytest.raises(ValueError):
        load_transcriptions(project)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("record_id", ""),
        ("equation", ""),
        ("equation", 2),
        ("pdf_page", []),
        ("section", ""),
        ("normalized_expression", ""),
        ("source_excerpt_path", ""),
        ("source_excerpt_path", 7),
        ("source_excerpt_byte_count", True),
        ("source_excerpt_byte_count", 0),
        ("source_excerpt_sha256", ""),
        ("source_excerpt_sha256", 7),
        ("reviewed_by", []),
        ("reviewed_by", "not a list"),
    ),
)
def test_rejects_wrong_or_empty_field_value(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    project = _copy_transcriptions(tmp_path)
    records = _read_manifest(project)
    records[0][field] = value
    _write_manifest(project, records)

    with pytest.raises((TypeError, ValueError)):
        load_transcriptions(project)


def test_rejects_reviewer_identity_mutation(tmp_path: Path) -> None:
    project = _copy_transcriptions(tmp_path)
    records = _read_manifest(project)
    records[0]["reviewed_by"] = [
        REVIEWERS[0],
        "independent-but-unapproved-reviewer",
    ]
    _write_manifest(project, records)

    with pytest.raises(ValueError):
        load_transcriptions(project)


@pytest.mark.parametrize(
    "bad_path",
    (
        "/tmp/eq-02.txt",
        "paper_transcriptions/excerpts/../algorithm1.txt",
    ),
)
def test_rejects_absolute_and_traversal_paths(
    tmp_path: Path,
    bad_path: str,
) -> None:
    project = _copy_transcriptions(tmp_path)
    records = _read_manifest(project)
    records[0]["source_excerpt_path"] = bad_path
    _write_manifest(project, records)

    with pytest.raises(ValueError):
        load_transcriptions(project)


def test_rejects_duplicate_record_id(tmp_path: Path) -> None:
    project = _copy_transcriptions(tmp_path)
    records = _read_manifest(project)
    records[1]["record_id"] = records[0]["record_id"]
    _write_manifest(project, records)

    with pytest.raises(ValueError):
        load_transcriptions(project)


def test_rejects_duplicate_logical_path(tmp_path: Path) -> None:
    project = _copy_transcriptions(tmp_path)
    records = _read_manifest(project)
    records[1]["source_excerpt_path"] = records[0]["source_excerpt_path"]
    records[1]["source_excerpt_byte_count"] = records[0][
        "source_excerpt_byte_count"
    ]
    records[1]["source_excerpt_sha256"] = records[0]["source_excerpt_sha256"]
    _write_manifest(project, records)

    with pytest.raises(ValueError):
        load_transcriptions(project)


def test_rejects_duplicate_json_key(tmp_path: Path) -> None:
    project = _copy_transcriptions(tmp_path)
    manifest_path = _manifest_path(project)
    text = manifest_path.read_text(encoding="utf-8")
    text = text.replace(
        '    "equation": "2",',
        '    "equation": "2",\n    "equation": "2",',
        1,
    )
    manifest_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_transcriptions(project)


def test_rejects_self_consistent_excerpt_mutation(tmp_path: Path) -> None:
    project = _copy_transcriptions(tmp_path)
    records = _read_manifest(project)
    excerpt = project / records[0]["source_excerpt_path"]
    tampered = b"self-consistent but unapproved transcription\n"
    excerpt.write_bytes(tampered)
    records[0]["source_excerpt_byte_count"] = len(tampered)
    records[0]["source_excerpt_sha256"] = hashlib.sha256(tampered).hexdigest()
    _write_manifest(project, records)

    with pytest.raises(ValueError):
        load_transcriptions(project)


@pytest.mark.parametrize("name", ("extra.txt", "extra.bin"))
def test_rejects_unreferenced_regular_file(
    tmp_path: Path,
    name: str,
) -> None:
    project = _copy_transcriptions(tmp_path)
    (project / "paper_transcriptions" / name).write_bytes(b"unreferenced")

    with pytest.raises(ValueError):
        load_transcriptions(project)


def test_rejects_transcription_root_symlink(tmp_path: Path) -> None:
    external = tmp_path / "external-transcriptions"
    shutil.copytree(PROJECT_ROOT / "paper_transcriptions", external)
    project = tmp_path / "project"
    project.mkdir()
    (project / "paper_transcriptions").symlink_to(
        external,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="symlink"):
        load_transcriptions(project)


def test_rejects_manifest_symlink_alias(tmp_path: Path) -> None:
    project = _copy_transcriptions(tmp_path)
    _replace_with_internal_symlink(_manifest_path(project))

    with pytest.raises(ValueError, match="symlink"):
        load_transcriptions(project)


def test_rejects_traversed_directory_symlink_alias(tmp_path: Path) -> None:
    project = _copy_transcriptions(tmp_path)
    excerpts = project / "paper_transcriptions" / "excerpts"
    _replace_with_internal_symlink(excerpts)

    with pytest.raises(ValueError, match="symlink"):
        load_transcriptions(project)


def test_rejects_excerpt_symlink_alias(tmp_path: Path) -> None:
    project = _copy_transcriptions(tmp_path)
    excerpt = project / "paper_transcriptions" / "excerpts" / "eq-02.txt"
    _replace_with_internal_symlink(excerpt)

    with pytest.raises(ValueError, match="symlink"):
        load_transcriptions(project)


def test_rejects_algorithm_symlink_alias(tmp_path: Path) -> None:
    project = _copy_transcriptions(tmp_path)
    algorithm = project / "paper_transcriptions" / "algorithm1.txt"
    _replace_with_internal_symlink(algorithm)

    with pytest.raises(ValueError, match="symlink"):
        load_transcriptions(project)


def test_rejects_unreferenced_symlink(tmp_path: Path) -> None:
    project = _copy_transcriptions(tmp_path)
    transcriptions = project / "paper_transcriptions"
    (transcriptions / "unreferenced-link").symlink_to("algorithm1.txt")

    with pytest.raises(ValueError, match="symlink"):
        load_transcriptions(project)
