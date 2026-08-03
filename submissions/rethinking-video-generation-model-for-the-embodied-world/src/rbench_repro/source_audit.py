import ast
from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath
import re
import shlex
from typing import Literal

from rbench_repro.acquisition import AcquiredSource
from rbench_repro.census import manifested_paths, read_manifested_bytes, shell_categories


TARGET_FAILURE_MODES = (
    "structural distortion",
    "floating components",
    "key-action omission",
)


@dataclass(frozen=True, slots=True)
class SourceLocation:
    source_label: str
    path: str
    start_line: int
    end_line: int
    span_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "end_line": self.end_line,
            "path": self.path,
            "source_label": self.source_label,
            "span_sha256": self.span_sha256,
            "start_line": self.start_line,
        }


@dataclass(frozen=True, slots=True)
class MetricTrace:
    partition: Literal["task", "embodiment"]
    identifier: str
    dimensions: tuple[str, ...]
    evaluator_path: str
    parser_path: str | None
    aggregation_path: str
    invoked_by_entry_point: bool
    source_locations: tuple[SourceLocation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "aggregation_path": self.aggregation_path,
            "dimensions": list(self.dimensions),
            "evaluator_path": self.evaluator_path,
            "identifier": self.identifier,
            "invoked_by_entry_point": self.invoked_by_entry_point,
            "parser_path": self.parser_path,
            "partition": self.partition,
            "source_locations": [item.to_dict() for item in self.source_locations],
        }


@dataclass(frozen=True, slots=True)
class FixtureOutcome:
    case: Literal["valid", "missing", "malformed", "boundary", "alias"]
    success: bool
    output: float | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "case": self.case,
            "error": self.error,
            "output": self.output,
            "success": self.success,
        }


@dataclass(frozen=True, slots=True)
class FailureModeResult:
    label: str
    aliases: tuple[str, ...]
    status: Literal["operationalized", "declared_only", "missing"]
    source_locations: tuple[SourceLocation, ...]
    evaluator_inputs: tuple[str, ...]
    output_type: str | None
    output_range: list[float] | None
    parser_path: str | None
    aggregation_path: str | None
    invoked_by_entry_point: bool
    fixtures: tuple[FixtureOutcome, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "aggregation_path": self.aggregation_path,
            "aliases": list(self.aliases),
            "evaluator_inputs": list(self.evaluator_inputs),
            "fixtures": [item.to_dict() for item in self.fixtures],
            "invoked_by_entry_point": self.invoked_by_entry_point,
            "label": self.label,
            "output_range": self.output_range,
            "output_type": self.output_type,
            "parser_path": self.parser_path,
            "source_locations": [item.to_dict() for item in self.source_locations],
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ParsedPython:
    source_label: str
    path: str
    payload: bytes
    tree: ast.Module


@dataclass(frozen=True, slots=True)
class ScalarReplica:
    input_alias: str
    lower: float
    upper: float


def trace_metrics(sources: dict[str, AcquiredSource]) -> tuple[MetricTrace, ...]:
    python_sources = parsed_python_sources(sources)
    shell_sources = source_payloads(sources, ".sh")
    traces = []
    for parsed in python_sources:
        partition = partition_for_path(parsed.path)
        if partition is None:
            continue
        identifier_assignment = next(
            (
                (node, name)
                for node in parsed.tree.body
                for name in assignment_names(node)
                if "identifier" in name.casefold()
                and isinstance(literal_value(node), str)
            ),
            None,
        )
        dimensions_assignment = next(
            (
                (node, name)
                for node in parsed.tree.body
                for name in assignment_names(node)
                if "dimension" in name.casefold()
                and string_sequence(literal_value(node)) is not None
            ),
            None,
        )
        if identifier_assignment is None or dimensions_assignment is None:
            continue
        identifier_node, identifier_name = identifier_assignment
        dimensions_node, dimensions_name = dimensions_assignment
        pipeline = metric_pipeline(parsed.tree, identifier_name, dimensions_name)
        if pipeline is None:
            continue
        parser, aggregate, coordinator = pipeline
        identifier = literal_value(identifier_node)
        dimensions = string_sequence(literal_value(dimensions_node))
        if not isinstance(identifier, str) or dimensions is None:
            continue
        locations = (
            node_location(parsed, identifier_node),
            node_location(parsed, dimensions_node),
            node_location(parsed, aggregate),
            node_location(parsed, coordinator),
        )
        traces.append(
            MetricTrace(
                partition=partition,
                identifier=identifier,
                dimensions=dimensions,
                evaluator_path=f"{parsed.source_label}:{parsed.path}",
                parser_path=f"{parsed.source_label}:{parsed.path}:{parser.name}",
                aggregation_path=f"{parsed.source_label}:{parsed.path}:{aggregate.name}",
                invoked_by_entry_point=shell_invokes(
                    shell_sources, parsed.source_label, parsed.path
                ),
                source_locations=locations,
            )
        )
    return tuple(sorted(traces, key=lambda trace: (trace.partition, trace.identifier)))


def audit_failure_modes(
    sources: dict[str, AcquiredSource],
) -> tuple[FailureModeResult, ...]:
    all_payloads = source_payloads(sources)
    python_sources = parsed_python_sources(sources)
    shell_sources = source_payloads(sources, ".sh")
    results = []
    for target in TARGET_FAILURE_MODES:
        locations = text_locations(all_payloads, target)
        aliases = set()
        route = None
        for parsed in python_sources:
            declared_aliases = exact_ast_aliases(parsed.tree, target)
            if not declared_aliases and not exact_ast_declaration(parsed.tree, target):
                continue
            aliases.update(declared_aliases)
            candidate = connected_failure_route(
                parsed, target, declared_aliases, shell_sources
            )
            if candidate is not None:
                route = candidate
        if not locations:
            results.append(missing_failure_mode(target))
            continue
        if route is None:
            results.append(
                FailureModeResult(
                    label=target,
                    aliases=tuple(sorted(aliases)),
                    status="declared_only",
                    source_locations=locations,
                    evaluator_inputs=(),
                    output_type=None,
                    output_range=None,
                    parser_path=None,
                    aggregation_path=None,
                    invoked_by_entry_point=False,
                    fixtures=(),
                )
            )
            continue
        path, parser_name, aggregate_name, replica = route
        results.append(
            FailureModeResult(
                label=target,
                aliases=tuple(sorted(aliases)),
                status="operationalized",
                source_locations=locations,
                evaluator_inputs=(replica.input_alias,),
                output_type="number",
                output_range=[replica.lower, replica.upper],
                parser_path=f"{path}:{parser_name}",
                aggregation_path=f"{path}:{aggregate_name}",
                invoked_by_entry_point=True,
                fixtures=run_scalar_fixtures(replica, target),
            )
        )
    return tuple(results)


def connected_failure_route(
    parsed: ParsedPython,
    target: str,
    aliases: set[str],
    shell_sources: tuple[tuple[str, str, bytes], ...],
) -> tuple[str, str, str, ScalarReplica] | None:
    identifiers = {target.replace("-", " ").replace(" ", "_")} | aliases
    functions = [node for node in parsed.tree.body if isinstance(node, ast.FunctionDef)]
    parser = next(
        (
            node
            for node in functions
            if "parse" in node.name.casefold()
            and any(identifier.casefold() in node.name.casefold() for identifier in identifiers)
        ),
        None,
    )
    aggregate = next(
        (
            node
            for node in functions
            if "aggregate" in node.name.casefold()
            and any(identifier.casefold() in node.name.casefold() for identifier in identifiers)
        ),
        None,
    )
    if parser is None or aggregate is None or not shell_invokes(
        shell_sources, parsed.source_label, parsed.path
    ):
        return None
    if connected_coordinator(parsed.tree, parser.name, aggregate.name) is None:
        return None
    replica = recover_scalar_replica(parser, aggregate, identifiers)
    if replica is None:
        return None
    return (
        f"{parsed.source_label}:{parsed.path}",
        parser.name,
        aggregate.name,
        replica,
    )


def run_scalar_fixtures(
    replica: ScalarReplica, target: str
) -> tuple[FixtureOutcome, ...]:
    cases = (
        ("valid", {replica.input_alias: "0.5"}),
        ("missing", {}),
        ("malformed", {replica.input_alias: "not-a-number"}),
        (
            "boundary",
            {
                replica.input_alias: str(
                    replica.upper + max(1.0, replica.upper - replica.lower)
                )
            },
        ),
    )
    outcomes = []
    for case, payload in cases:
        try:
            value = float(payload[replica.input_alias])
            bounded = min(replica.upper, max(replica.lower, value))
            output = sum([bounded]) / len([bounded])
        except KeyError:
            outcomes.append(FixtureOutcome(case, False, None, "missing value"))
        except (TypeError, ValueError):
            outcomes.append(FixtureOutcome(case, False, None, "invalid number"))
        else:
            outcomes.append(FixtureOutcome(case, True, output, None))
    if replica.input_alias.casefold() == target.casefold():
        outcomes.append(
            FixtureOutcome("alias", False, None, "distinct alias unavailable")
        )
    else:
        value = min(replica.upper, max(replica.lower, float("0.25")))
        outcomes.append(FixtureOutcome("alias", True, value, None))
    return tuple(outcomes)


def missing_failure_mode(target: str) -> FailureModeResult:
    return FailureModeResult(
        label=target,
        aliases=(),
        status="missing",
        source_locations=(),
        evaluator_inputs=(),
        output_type=None,
        output_range=None,
        parser_path=None,
        aggregation_path=None,
        invoked_by_entry_point=False,
        fixtures=(),
    )


def source_payloads(
    sources: dict[str, AcquiredSource], suffix: str | None = None
) -> tuple[tuple[str, str, bytes], ...]:
    payloads = []
    for source in sources.values():
        for path in sorted(manifested_paths(source)):
            if suffix is None or path.endswith(suffix):
                payloads.append(
                    (source.manifest.label, path, read_manifested_bytes(source, path))
                )
    return tuple(payloads)


def parsed_python_sources(sources: dict[str, AcquiredSource]) -> tuple[ParsedPython, ...]:
    parsed = []
    for source_label, path, payload in source_payloads(sources, ".py"):
        try:
            tree = ast.parse(payload)
        except (SyntaxError, UnicodeDecodeError):
            continue
        parsed.append(ParsedPython(source_label, path, payload, tree))
    return tuple(parsed)


def text_locations(
    payloads: tuple[tuple[str, str, bytes], ...], target: str
) -> tuple[SourceLocation, ...]:
    pattern = re.compile(rf"(?<!\w){re.escape(target)}(?!\w)", re.IGNORECASE)
    locations = []
    for _source_label, path, payload in payloads:
        for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
            if pattern.search(line.decode(errors="replace")):
                locations.append(
                    SourceLocation(
                        source_label=_source_label,
                        path=path,
                        start_line=line_number,
                        end_line=line_number,
                        span_sha256=sha256(line).hexdigest(),
                    )
                )
    return tuple(sorted(locations, key=lambda item: (item.path, item.start_line)))


def exact_ast_declaration(tree: ast.AST, target: str) -> bool:
    return any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and literal_contains_target(literal_value(node), target)
        for node in ast.walk(tree)
    )


def literal_contains_target(value: object, target: str) -> bool:
    if isinstance(value, str):
        return value.casefold().strip() == target
    if isinstance(value, dict):
        return any(
            literal_contains_target(item, target)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (list, tuple, set)):
        return any(literal_contains_target(item, target) for item in value)
    return False


def exact_ast_aliases(tree: ast.AST, target: str) -> set[str]:
    aliases = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and value.value.casefold().strip() == target
            ):
                aliases.add(key.value)
    return aliases


def shell_invokes(
    shell_sources: tuple[tuple[str, str, bytes], ...],
    source_label: str,
    evaluator_path: str,
) -> bool:
    for shell_source_label, _path, payload in shell_sources:
        if shell_source_label != source_label:
            continue
        text = payload.decode(errors="replace")
        for line in text.splitlines():
            try:
                tokens = shlex.split(line, comments=True, posix=True)
            except ValueError:
                continue
            if python_executes(tokens, evaluator_path):
                return True
        template = "eval/5_tasks/${TASK_TYPE}.py"
        if evaluator_path.startswith("eval/5_tasks/") and (
            any(
                python_executes(shell_tokens(line), template)
                for line in text.splitlines()
            )
            and PurePosixPath(evaluator_path).stem in shell_categories(text, "task")
        ):
            return True
    return False


def shell_tokens(line: str) -> list[str]:
    try:
        return shlex.split(line, comments=True, posix=True)
    except ValueError:
        return []


def python_executes(tokens: list[str], evaluator_path: str) -> bool:
    return (
        len(tokens) >= 2
        and PurePosixPath(tokens[0]).name in {"python", "python3"}
        and tokens[1] == evaluator_path
    )


def partition_for_path(path: str) -> Literal["task", "embodiment"] | None:
    if path.startswith("eval/5_tasks/"):
        return "task"
    if path.startswith("eval/4_embodiments/"):
        return "embodiment"
    return None


def assignment_names(node: ast.AST) -> tuple[str, ...]:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return ()
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return tuple(
        target.id for target in targets if isinstance(target, ast.Name)
    )


def literal_value(node: ast.Assign | ast.AnnAssign) -> object:
    try:
        return ast.literal_eval(node.value)
    except (ValueError, TypeError):
        return None


def string_sequence(value: object) -> tuple[str, ...] | None:
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return None


def node_location(parsed: ParsedPython, node: ast.AST) -> SourceLocation:
    start = node.lineno
    end = node.end_lineno or start
    lines = parsed.payload.splitlines(keepends=True)
    span = b"".join(lines[start - 1 : end])
    return SourceLocation(
        parsed.source_label, parsed.path, start, end, sha256(span).hexdigest()
    )


def metric_pipeline(
    tree: ast.Module, identifier_name: str, dimensions_name: str
) -> tuple[ast.FunctionDef, ast.FunctionDef, ast.FunctionDef] | None:
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    parsers = [node for node in functions if "parse" in node.name.casefold()]
    aggregators = [node for node in functions if "aggregate" in node.name.casefold()]
    for parser in parsers:
        for aggregate in aggregators:
            coordinator = connected_coordinator(tree, parser.name, aggregate.name)
            if coordinator is None:
                continue
            if returns_metric_structure(
                coordinator, identifier_name, dimensions_name, aggregate.name
            ):
                return parser, aggregate, coordinator
    return None


def returns_metric_structure(
    coordinator: ast.FunctionDef,
    identifier_name: str,
    dimensions_name: str,
    aggregate_name: str,
) -> bool:
    for returned in coordinator.body:
        if not isinstance(returned, ast.Return) or not isinstance(returned.value, ast.Dict):
            continue
        dictionary = returned.value
        dimensions_present = any(
            isinstance(key, ast.Constant)
            and key.value == "dimensions"
            and isinstance(value, ast.Name)
            and value.id == dimensions_name
            for key, value in zip(dictionary.keys, dictionary.values, strict=True)
        )
        metric_present = any(
            isinstance(key, ast.Name)
            and key.id == identifier_name
            and calls_name(value, aggregate_name)
            for key, value in zip(dictionary.keys, dictionary.values, strict=True)
        )
        if dimensions_present and metric_present:
            return True
    return False


def connected_coordinator(
    tree: ast.Module, parser_name: str, aggregate_name: str
) -> ast.FunctionDef | None:
    for function in (
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name not in {parser_name, aggregate_name}
    ):
        for return_index, returned in enumerate(function.body):
            if not isinstance(returned, ast.Return) or returned.value is None:
                continue
            returned_calls = (
                node
                for node in ast.walk(returned.value)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == aggregate_name
            )
            for call in returned_calls:
                if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
                    continue
                variable = call.args[0].id
                assignments = [
                    statement
                    for statement in function.body[:return_index]
                    if variable in assignment_names(statement)
                ]
                if assignments and parser_derived_value(
                    assignments[-1].value, parser_name
                ):
                    return function
    return None


def calls_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == name
        for candidate in ast.walk(node)
    )


def parser_derived_value(node: ast.AST, parser_name: str) -> bool:
    if is_named_call(node, parser_name):
        return True
    return (
        isinstance(node, ast.ListComp)
        and is_named_call(node.elt, parser_name)
    )


def recover_scalar_replica(
    parser: ast.FunctionDef, aggregate: ast.FunctionDef, identifiers: set[str]
) -> ScalarReplica | None:
    parsed = exact_scalar_parser(parser)
    if parsed is None:
        return None
    input_alias, lower, upper = parsed
    if input_alias not in identifiers or not is_mean_aggregator(aggregate):
        return None
    return ScalarReplica(input_alias, lower, upper)


def exact_scalar_parser(parser: ast.FunctionDef) -> tuple[str, float, float] | None:
    if len(parser.args.args) != 1 or len(parser.body) != 2:
        return None
    assignment, returned = parser.body
    if (
        not isinstance(assignment, ast.Assign)
        or len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], ast.Name)
        or not isinstance(assignment.value, ast.Subscript)
        or not isinstance(assignment.value.value, ast.Name)
        or assignment.value.value.id != parser.args.args[0].arg
        or not isinstance(assignment.value.slice, ast.Constant)
        or not isinstance(assignment.value.slice.value, str)
        or not isinstance(returned, ast.Return)
    ):
        return None
    expression = returned.value
    if not is_named_call(expression, "min") or len(expression.args) != 2:
        return None
    upper = numeric_constant(expression.args[0])
    maximum = expression.args[1]
    if upper is None or not is_named_call(maximum, "max") or len(maximum.args) != 2:
        return None
    lower = numeric_constant(maximum.args[0])
    conversion = maximum.args[1]
    if (
        lower is None
        or lower > upper
        or not is_named_call(conversion, "float")
        or len(conversion.args) != 1
        or not isinstance(conversion.args[0], ast.Name)
        or conversion.args[0].id != assignment.targets[0].id
    ):
        return None
    return assignment.value.slice.value, lower, upper


def is_mean_aggregator(function: ast.FunctionDef) -> bool:
    if len(function.args.args) != 1 or len(function.body) != 1:
        return False
    returned = function.body[0]
    if (
        not isinstance(returned, ast.Return)
        or not isinstance(returned.value, ast.BinOp)
        or not isinstance(returned.value.op, ast.Div)
    ):
        return False
    argument = function.args.args[0].arg
    return named_single_argument_call(returned.value.left, "sum", argument) and (
        named_single_argument_call(returned.value.right, "len", argument)
    )


def named_single_argument_call(node: ast.AST, name: str, argument: str) -> bool:
    return (
        is_named_call(node, name)
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == argument
    )


def is_named_call(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    )


def numeric_constant(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return float(node.value)
    return None
