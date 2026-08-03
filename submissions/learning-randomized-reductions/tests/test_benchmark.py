from pathlib import Path
import pytest

from lrr_repro.benchmark import (
    build_census,
    extract_test_ids,
    read_primary_csv_rows,
)


def base_text(project_root: Path) -> str:
    return (
        project_root
        / "evidence/inputs/upstream/src/bitween/evaluation/evaluation_rsr_bench_paper.py"
    ).read_text(encoding="utf-8")


def extended_text(project_root: Path) -> str:
    return (
        project_root
        / "evidence/inputs/upstream/src/bitween/evaluation/evaluation_rsr_bench_paper_extended.py"
    ).read_text(encoding="utf-8")


def csv_path(project_root: Path) -> Path:
    return (
        project_root
        / "evidence/inputs/upstream/results/Bitween-Results(Sheet1-ICML).csv"
    )


def test_pinned_sources_and_csv_reconcile_to_exactly_80(project_root):
    records = build_census(
        base_text(project_root), extended_text(project_root), csv_path(project_root)
    )
    assert [record.benchmark_id for record in records] == list(range(1, 81))
    assert len({record.csv_name for record in records}) == 80
    assert records[32].csv_name == "sigmoid"


def test_continuation_rows_do_not_inflate_count(project_root):
    assert len(read_primary_csv_rows(csv_path(project_root))) == 80


def test_duplicate_or_missing_id_fails(tmp_path, project_root):
    # Create corrupted CSV with missing 80
    lines = csv_path(project_root).read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("80,"):
            new_lines.append(line.replace("80,", "79,", 1))
        else:
            new_lines.append(line)
    corrupt_csv = tmp_path / "corrupt.csv"
    corrupt_csv.write_text("\n".join(new_lines), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 1..80"):
        build_census(base_text(project_root), extended_text(project_root), corrupt_csv)
