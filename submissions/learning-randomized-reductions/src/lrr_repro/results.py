"""Results processing, aggregation, and algebra verification module."""

import csv
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

import sympy


@dataclass(frozen=True)
class BackendSummary:
    backend: str
    rsr_total: int
    covered_benchmarks: int
    coverage: Fraction
    runtime_min: Decimal
    runtime_mean: Decimal
    runtime_max: Decimal


def summarize_backend(
    rows: Sequence[Mapping[str, str]],
    rsr_column: int,
    time_column: int,
    name: str,
) -> BackendSummary:
    rsr_total = 0
    covered_count = 0
    times: list[Decimal] = []

    for row in rows:
        rsr_str = row.get(str(rsr_column), "0").strip()
        val = int(rsr_str) if rsr_str.isdigit() else 0
        if val > 0:
            covered_count += 1
        rsr_total += val

        t_str = row.get(str(time_column), "").strip()
        if t_str:
            try:
                times.append(Decimal(t_str))
            except Exception:
                pass

    coverage = Fraction(covered_count, len(rows))
    r_min = min(times) if times else Decimal("0")
    r_max = max(times) if times else Decimal("0")
    r_sum = sum(times) if times else Decimal("0")
    r_mean = (r_sum / Decimal(len(times))).quantize(Decimal("0.001")) if times else Decimal("0")

    return BackendSummary(
        backend=name,
        rsr_total=rsr_total,
        covered_benchmarks=covered_count,
        coverage=coverage,
        runtime_min=r_min,
        runtime_mean=r_mean,
        runtime_max=r_max,
    )


def verify_sigmoid_identity() -> bool:
    x, r = sympy.symbols("x r", real=True)
    sigma = lambda val: 1 / (1 + sympy.exp(-val))
    rhs = sigma(x + r) * (sigma(r) - 1) / (
        2 * sigma(x + r) * sigma(r) - sigma(x + r) - sigma(r)
    )
    lhs = sigma(x)
    diff = sympy.simplify(lhs - rhs)
    return bool(diff == 0)


def extract_function_arguments(expression: str) -> tuple[str, ...]:
    args = []
    idx = 0
    length = len(expression)
    while idx < length:
        pos = expression.find("f(", idx)
        if pos == -1:
            break
        # Make sure 'f' is not part of an identifier like 'of(' or 'coeff('
        if pos > 0 and (expression[pos - 1].isalnum() or expression[pos - 1] == "_"):
            idx = pos + 2
            continue

        start_arg = pos + 2
        depth = 1
        curr = start_arg
        while curr < length and depth > 0:
            if expression[curr] == "(":
                depth += 1
            elif expression[curr] == ")":
                depth -= 1
            curr += 1
        if depth == 0:
            arg = expression[start_arg : curr - 1]
            args.append(arg)
            idx = curr
        else:
            idx = start_arg
    return tuple(args)


def canonical_query(argument: str) -> str:
    cleaned = argument.replace(" ", "")
    if cleaned in ("r+x", "x+r"):
        return "x+r"
    if cleaned in ("r*x", "x*r"):
        return "x*r"
    return cleaned


def novel_agentic_queries(csv_path: Path) -> tuple[str, ...]:
    fixed_set = {"x+r", "x-r", "x*r", "x", "r"}
    found_queries = set()

    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            # Inspect only Claude-Opus-4.1 Agentic verified property field (column 54)
            for col in (54,):
                if col < len(row):
                    cell = row[col]
                    if "f(" in cell:
                        raw_args = extract_function_arguments(cell)
                        for arg in raw_args:
                            can = canonical_query(arg)
                            if can and can not in fixed_set:
                                found_queries.add(can)


    return tuple(sorted(found_queries))
