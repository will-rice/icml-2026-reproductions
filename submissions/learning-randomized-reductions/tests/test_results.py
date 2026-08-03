import csv
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import pytest

from lrr_repro.benchmark import read_primary_csv_rows
from lrr_repro.results import (
    canonical_query,
    extract_function_arguments,
    novel_agentic_queries,
    summarize_backend,
    verify_sigmoid_identity,
)


def csv_path(project_root: Path) -> Path:
    return (
        project_root
        / "evidence/inputs/upstream/results/Bitween-Results(Sheet1-ICML).csv"
    )


def test_recomputes_vanilla_and_agentic_coverage(project_root):
    primary_rows = read_primary_csv_rows(csv_path(project_root))
    lr = summarize_backend(primary_rows, 18, 21, "vanilla-lr")
    agentic = summarize_backend(primary_rows, 53, 56, "agentic-opus")

    assert (lr.rsr_total, lr.covered_benchmarks, lr.coverage) == (
        87,
        43,
        Fraction(43, 80),
    )
    assert (lr.runtime_min, lr.runtime_mean, lr.runtime_max) == (
        Decimal("0.13"),
        Decimal("4.791"),
        Decimal("19.12"),
    )
    assert (agentic.rsr_total, agentic.covered_benchmarks, agentic.coverage) == (
        793,
        64,
        Fraction(4, 5),
    )


def test_sigmoid_reduction_is_an_identity():
    assert verify_sigmoid_identity() is True


def test_agentic_outputs_contain_queries_outside_fixed_set(project_root):
    queries = novel_agentic_queries(csv_path(project_root))
    assert "x+log(k)" in queries
    assert not {"x+r", "x-r", "x*r", "x", "r"}.intersection(queries)


def test_novel_agentic_queries_ignores_non_agentic_columns(tmp_path):
    csv_file = tmp_path / "test_results.csv"
    # Create row with 60 columns where x+log(k) is in col 5 (pysr), not in cols 53..55 (agentic opus)
    row = [""] * 60
    row[0] = "1"
    row[5] = "Eq(f(x+log(k)), 0)"
    csv_file.write_text(",".join(row) + "\n", encoding="utf-8")

    queries = novel_agentic_queries(csv_file)
    assert "x+log(k)" not in queries


def test_novel_agentic_queries_excludes_unverified_properties(tmp_path):
    csv_file = tmp_path / "test_unverified.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        row = [""] * 58
        row[0] = "1"
        row[54] = "Eq(f(x+r) - f(x), 0)"
        row[55] = "Eq(f(x+log(99)) - 1, 0)"
        writer.writerow(row)

    queries = novel_agentic_queries(csv_file)
    assert "x+log(99)" not in queries
