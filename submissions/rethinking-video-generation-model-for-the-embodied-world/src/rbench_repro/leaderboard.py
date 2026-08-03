import ast
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, ROUND_HALF_UP
from hashlib import sha256
import json
from typing import Literal, cast

from rbench_repro.acquisition import AcquiredSource
from rbench_repro.census import CATEGORY_RULES, category_name, read_manifested_bytes


@dataclass(frozen=True, slots=True)
class Formula:
    columns: tuple[str, ...]
    operation: Literal["mean"]
    source_precision: int
    output_precision: int
    rounding: Literal["half_even", "half_up"]
    absolute_tolerance: str


@dataclass(frozen=True, slots=True)
class LeaderboardResult:
    cohort: Literal["paper-era", "later"]
    filename: str
    raw_count: int
    valid_count: int
    unique_exact_count: int
    unique_normalized_count: int
    ordered_names: tuple[str, ...]
    ordered_name_hash: str
    duplicate_names: tuple[str, ...]
    normalization_collisions: tuple[tuple[str, ...], ...]
    invalid_rows: tuple[dict[str, object], ...]
    rows: tuple[dict[str, object], ...]
    discrepancies: tuple[dict[str, object], ...]
    group_orphans: tuple[str, ...]
    multiply_assigned_models: tuple[str, ...]
    unassigned_models: tuple[str, ...]
    records: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "cohort": self.cohort,
            "discrepancies": [dict(item) for item in self.discrepancies],
            "duplicate_names": list(self.duplicate_names),
            "filename": self.filename,
            "group_orphans": list(self.group_orphans),
            "invalid_rows": [dict(item) for item in self.invalid_rows],
            "multiply_assigned_models": list(self.multiply_assigned_models),
            "normalization_collisions": [
                list(group) for group in self.normalization_collisions
            ],
            "ordered_name_hash": self.ordered_name_hash,
            "ordered_names": list(self.ordered_names),
            "raw_count": self.raw_count,
            "rows": [dict(item) for item in self.rows],
            "unassigned_models": list(self.unassigned_models),
            "unique_exact_count": self.unique_exact_count,
            "unique_normalized_count": self.unique_normalized_count,
            "valid_count": self.valid_count,
        }


@dataclass(frozen=True, slots=True)
class CohortComparison:
    added_models: tuple[str, ...]
    prepended_models: tuple[str, ...]
    removed_models: tuple[str, ...]
    reordered_models: tuple[str, ...]
    shared_records_field_equal: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "added_models": list(self.added_models),
            "prepended_models": list(self.prepended_models),
            "removed_models": list(self.removed_models),
            "reordered_models": list(self.reordered_models),
            "shared_records_field_equal": self.shared_records_field_equal,
        }


def derive_formula(utils_source: bytes, app_source: bytes) -> Formula:
    try:
        trees = (ast.parse(utils_source), ast.parse(app_source))
    except (SyntaxError, UnicodeDecodeError):
        raise ValueError("untraceable leaderboard formula") from None

    expected = tuple(rule.name for rule in CATEGORY_RULES)
    formulas = set()
    for tree in trees:
        column_assignments = category_column_assignments(tree, expected)
        traced = trace_formula_precision(tree, frozenset(column_assignments))
        if traced is not None:
            column_name, source_precision, output_precision = traced
            formulas.add(
                (column_assignments[column_name], source_precision, output_precision)
            )

    if len(formulas) != 1:
        raise ValueError("untraceable leaderboard formula")
    columns, source_precision, output_precision = formulas.pop()
    tolerance = Decimal(5).scaleb(-(output_precision + 1))
    return Formula(
        columns=columns,
        operation="mean",
        source_precision=source_precision,
        output_precision=output_precision,
        rounding="half_even",
        absolute_tolerance=format(tolerance, "f"),
    )


def infer_displayed_mean_formula(
    records: object, columns: tuple[str, ...]
) -> Formula:
    """Accept a mean rule only when committed display values are consistent.

    The extra half-unit of tolerance accounts for the fact that both the
    dimension values and the aggregate are available only at display
    precision. This establishes an artifact-consistency rule, not an upstream
    source-traced aggregation formula.
    """
    formula = Formula(
        columns=columns,
        operation="mean",
        source_precision=3,
        output_precision=3,
        rounding="half_even",
        absolute_tolerance="0.001",
    )
    if not isinstance(records, list) or not records:
        raise ValueError("invalid leaderboard records")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid leaderboard records")
        numeric = {
            column: decimal_score(record.get(column))
            for column in (*columns, "avg")
        }
        if any(value is None for value in numeric.values()):
            raise ValueError("invalid leaderboard records")
        values = cast(dict[str, Decimal], numeric)
        if (
            abs(values["avg"] - recompute(values, formula))
            > Decimal(formula.absolute_tolerance)
        ):
            raise ValueError("inconsistent displayed mean")
    return formula


def derive_groups(app_source: bytes) -> dict[str, tuple[str, ...]]:
    try:
        tree = ast.parse(app_source)
    except (SyntaxError, UnicodeDecodeError):
        raise ValueError("untraceable leaderboard groups") from None
    groups = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        for name in names:
            if name == "GROUPS":
                try:
                    mapping = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                if isinstance(mapping, dict) and all(
                    isinstance(label, str)
                    and isinstance(members, (list, tuple))
                    and all(isinstance(member, str) for member in members)
                    for label, members in mapping.items()
                ):
                    groups.update(
                        {
                            label.casefold(): tuple(members)
                            for label, members in mapping.items()
                        }
                    )
                continue
            if not name.casefold().endswith("_models"):
                continue
            try:
                members = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(members, (list, tuple)) and all(
                isinstance(member, str) for member in members
            ):
                groups[name[: -len("_models")].casefold()] = tuple(members)
    if not groups:
        raise ValueError("untraceable leaderboard groups")
    return dict(sorted(groups.items()))


def audit_leaderboard(
    source: AcquiredSource,
    cohort: Literal["paper-era", "later"],
    filename: str,
    formula: Formula,
    groups: dict[str, tuple[str, ...]],
) -> LeaderboardResult:
    try:
        value = json.loads(read_manifested_bytes(source, filename))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("invalid leaderboard JSON") from None
    if not isinstance(value, list):
        raise ValueError("invalid leaderboard schema")

    names = []
    valid_records = []
    audited_rows = []
    invalid_rows = []
    discrepancies = []
    valid_count = 0
    for index, raw in enumerate(value):
        errors = []
        if not isinstance(raw, dict):
            invalid_rows.append({"errors": ["row is not an object"], "index": index})
            continue
        model = raw.get("model")
        if not isinstance(model, str) or not model.strip():
            errors.append("invalid model name")
        missing = [column for column in formula.columns if column not in raw]
        if missing:
            errors.append("missing dimension columns")
        expected_fields = {"model", "avg", *formula.columns}
        if "avg" not in raw:
            errors.append("missing average")
        if set(raw) - expected_fields:
            errors.append("unexpected columns")

        numeric = {}
        for column in (*formula.columns, "avg"):
            if column not in raw:
                continue
            parsed = decimal_score(raw[column])
            if parsed is None:
                errors.append(f"invalid numeric value: {column}")
            else:
                numeric[column] = parsed
        if errors:
            invalid_rows.append({"errors": errors, "index": index, "model": model})
            continue

        valid_count += 1
        names.append(model)
        valid_records.append(raw)
        recomputed = recompute(numeric, formula)
        actual = numeric["avg"]
        absolute_error = abs(actual - recomputed)
        row = {
            "index": index,
            "model": model,
            "recomputed_avg": fixed_decimal(recomputed, formula.output_precision),
        }
        audited_rows.append(row)
        if absolute_error > Decimal(formula.absolute_tolerance):
            discrepancies.append(
                {
                    "absolute_error": fixed_decimal(absolute_error, formula.output_precision),
                    "index": index,
                    "model": model,
                    "reported_avg": fixed_decimal(actual, formula.output_precision),
                    "recomputed_avg": fixed_decimal(recomputed, formula.output_precision),
                }
            )

    ordered_names = tuple(names)
    exact_counts = {name: ordered_names.count(name) for name in set(ordered_names)}
    normalized_groups: dict[str, set[str]] = {}
    for name in ordered_names:
        normalized_groups.setdefault(normalized_name(name), set()).add(name)
    collisions = tuple(
        tuple(sorted(group))
        for group in sorted(normalized_groups.values(), key=lambda group: sorted(group))
        if len(group) > 1
    )
    memberships: dict[str, set[str]] = {}
    for group, members in groups.items():
        for member in members:
            memberships.setdefault(normalized_name(member), set()).add(group)
    actual_normalized = {normalized_name(name) for name in ordered_names}
    declared_names = {member for members in groups.values() for member in members}
    group_orphans = tuple(
        sorted(name for name in declared_names if normalized_name(name) not in actual_normalized)
    )
    return LeaderboardResult(
        cohort=cohort,
        filename=filename,
        raw_count=len(value),
        valid_count=valid_count,
        unique_exact_count=len(set(ordered_names)),
        unique_normalized_count=len(set(map(normalized_name, ordered_names))),
        ordered_names=ordered_names,
        ordered_name_hash=ordered_name_hash(ordered_names),
        duplicate_names=tuple(sorted(name for name, count in exact_counts.items() if count > 1)),
        normalization_collisions=collisions,
        invalid_rows=tuple(invalid_rows),
        rows=tuple(audited_rows),
        discrepancies=tuple(discrepancies),
        group_orphans=group_orphans,
        multiply_assigned_models=tuple(
            name
            for name in ordered_names
            if len(memberships.get(normalized_name(name), ())) > 1
        ),
        unassigned_models=tuple(
            name for name in ordered_names if normalized_name(name) not in memberships
        ),
        records=tuple(valid_records),
    )


def compare_cohorts(
    paper: LeaderboardResult, later: LeaderboardResult
) -> CohortComparison:
    paper_names = paper.ordered_names
    later_names = later.ordered_names
    paper_set = set(paper_names)
    later_set = set(later_names)
    added = tuple(name for name in later_names if name not in paper_set)
    removed = tuple(name for name in paper_names if name not in later_set)
    first_shared = next(
        (index for index, name in enumerate(later_names) if name in paper_set), len(later_names)
    )
    prepended = tuple(name for name in later_names[:first_shared] if name in added)
    expected_shared = tuple(name for name in paper_names if name in later_set)
    actual_shared = tuple(name for name in later_names if name in paper_set)
    reordered = () if expected_shared == actual_shared else actual_shared
    paper_by_name = {record.get("model"): record for record in paper.records}
    later_by_name = {record.get("model"): record for record in later.records}
    shared_equal = not paper.duplicate_names and not later.duplicate_names and all(
        paper_by_name[name] == later_by_name[name] for name in paper_set & later_set
    )
    return CohortComparison(
        added_models=added,
        prepended_models=prepended,
        removed_models=removed,
        reordered_models=reordered,
        shared_records_field_equal=shared_equal,
    )


def recompute(row: dict[str, Decimal], formula: Formula) -> Decimal:
    source_quantum = Decimal(1).scaleb(-formula.source_precision)
    rounding = ROUND_HALF_EVEN if formula.rounding == "half_even" else ROUND_HALF_UP
    values = tuple(row[column].quantize(source_quantum, rounding=rounding) for column in formula.columns)
    mean = sum(values, Decimal(0)) / Decimal(len(values))
    return mean.quantize(Decimal(1).scaleb(-formula.output_precision), rounding=rounding)


def normalized_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def decimal_score(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        parsed = Decimal(str(value))
    except InvalidOperation:
        return None
    if not parsed.is_finite() or parsed < 0 or parsed > 100:
        return None
    return parsed


def ordered_name_hash(names: tuple[str, ...]) -> str:
    payload = json.dumps(names, ensure_ascii=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def fixed_decimal(value: Decimal, precision: int) -> str:
    return f"{value:.{precision}f}"


def is_round_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "round"
        and len(node.args) == 2
    )


def is_mean_expression(node: ast.AST) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
        return False
    calls = [child for child in ast.walk(node) if isinstance(child, ast.Call)]
    names = {
        call.func.id for call in calls if isinstance(call.func, ast.Name)
    }
    return {"sum", "len"} <= names


def category_column_assignments(
    tree: ast.AST, expected: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    assignments = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        try:
            candidate = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        if (
            not isinstance(candidate, (list, tuple))
            or not all(isinstance(item, str) for item in candidate)
            or tuple(category_name(item) for item in candidate) != expected
        ):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = tuple(candidate)
    return assignments


def trace_formula_precision(
    tree: ast.AST, column_names: frozenset[str]
) -> tuple[str, int, int] | None:
    for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
        assignments = {
            target.id: node.value
            for node in function.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }
        for call in (node for node in ast.walk(function) if is_round_call(node)):
            output = integer_precision(call)
            if output is None or not is_mean_expression(call.args[0]):
                continue
            connected = [call.args[0]]
            connected.extend(
                assignments[name.id]
                for name in ast.walk(call.args[0])
                if isinstance(name, ast.Name) and name.id in assignments
            )
            connected_columns = {
                name.id
                for expression in connected
                for name in ast.walk(expression)
                if isinstance(name, ast.Name) and name.id in column_names
            }
            source_precisions = {
                precision
                for expression in connected
                for candidate in ast.walk(expression)
                if is_round_call(candidate)
                and candidate is not call
                and any(
                    isinstance(child, ast.Subscript)
                    for child in ast.walk(candidate.args[0])
                )
                if (precision := integer_precision(candidate)) is not None
            }
            if len(connected_columns) == 1 and len(source_precisions) == 1:
                return connected_columns.pop(), source_precisions.pop(), output
    return None


def integer_precision(call: ast.Call) -> int | None:
    precision = call.args[1]
    if isinstance(precision, ast.Constant) and type(precision.value) is int:
        return precision.value
    return None
