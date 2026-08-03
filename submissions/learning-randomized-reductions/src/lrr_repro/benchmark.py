"""80-function RSR-Bench census lane."""

import ast
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkRecord:
    benchmark_id: int
    source_name: str
    csv_name: str


def extract_test_ids(source: str) -> tuple[tuple[int, str], ...]:
    tree = ast.parse(source)
    results = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "test_id":
                    val = None
                    if isinstance(keyword.value, ast.Constant) and isinstance(
                        keyword.value.value, str
                    ):
                        val = keyword.value.value
                    if val and "_" in val:
                        parts = val.split("_", 1)
                        if parts[0].isdigit():
                            b_id = int(parts[0])
                            name = parts[1]
                            results.append((b_id, name))
            self.generic_visit(node)

    Visitor().visit(tree)
    return tuple(results)


def read_primary_csv_rows(path: Path) -> tuple[dict[str, str], ...]:
    primary_rows = []
    with open(path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            col0 = row[0].strip()
            if col0.isdigit():
                b_id = int(col0)
                if 1 <= b_id <= 80:
                    row_dict = {str(idx): val.strip() for idx, val in enumerate(row)}
                    row_dict["primary_id"] = col0
                    primary_rows.append(row_dict)
    return tuple(primary_rows)


def build_census(
    base_source: str, extended_source: str, csv_path: Path
) -> tuple[BenchmarkRecord, ...]:
    base_ids = dict(extract_test_ids(base_source))
    ext_ids = dict(extract_test_ids(extended_source))

    source_map: dict[int, str] = {}
    for b_id, name in base_ids.items():
        if 1 <= b_id <= 40:
            source_map[b_id] = name
    for b_id, name in ext_ids.items():
        if 41 <= b_id <= 80:
            source_map[b_id] = name

    primary_rows = read_primary_csv_rows(csv_path)
    if len(primary_rows) != 80:
        raise ValueError(
            f"Expected exactly 80 primary CSV rows (1..80), got {len(primary_rows)}"
        )

    csv_map: dict[int, str] = {}
    for row in primary_rows:
        b_id = int(row["primary_id"])
        csv_name = row.get("1", "").strip()
        if b_id in csv_map:
            raise ValueError(f"Duplicate CSV primary ID: {b_id} (expected exactly 1..80)")
        csv_map[b_id] = csv_name

    records = []
    for b_id in range(1, 81):
        if b_id not in source_map:
            raise ValueError(
                f"Missing source benchmark id {b_id} (expected exactly 1..80)"
            )
        if b_id not in csv_map:
            raise ValueError(
                f"Missing CSV benchmark id {b_id} (expected exactly 1..80)"
            )
        records.append(
            BenchmarkRecord(
                benchmark_id=b_id,
                source_name=source_map[b_id],
                csv_name=csv_map[b_id],
            )
        )

    return tuple(records)
