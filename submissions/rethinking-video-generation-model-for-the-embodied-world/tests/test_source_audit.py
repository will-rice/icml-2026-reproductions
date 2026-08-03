from hashlib import sha256

from conftest import acquired_source
from rbench_repro.source_audit import audit_failure_modes, trace_metrics


def test_metric_trace_connects_identifiers_dimensions_and_entry_points(
    source_audit_fixture,
):
    traces = [trace.to_dict() for trace in trace_metrics(source_audit_fixture)]
    assert {trace["partition"] for trace in traces} == {"task", "embodiment"}
    assert all(trace["identifier"] and trace["aggregation_path"] for trace in traces)
    assert all(trace["invoked_by_entry_point"] for trace in traces)
    assert {len(trace["dimensions"]) for trace in traces} == {4, 5}


def test_failure_modes_distinguish_connected_declared_and_missing(source_audit_fixture):
    results = {result.label: result for result in audit_failure_modes(source_audit_fixture)}
    assert results["structural distortion"].status == "operationalized"
    assert results["floating components"].status == "declared_only"
    assert results["key-action omission"].status == "missing"
    assert results["structural distortion"].source_locations[0].span_sha256


def test_failure_mode_fixtures_cover_valid_missing_malformed_boundary_and_alias(
    source_audit_fixture,
):
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    assert {outcome.case for outcome in result.fixtures} == {
        "valid",
        "missing",
        "malformed",
        "boundary",
        "alias",
    }
    assert result.output_type == "number"
    assert result.output_range == [0.0, 1.0]
    assert all(outcome.error is not None for outcome in result.fixtures if not outcome.success)
    alias = next(outcome for outcome in result.fixtures if outcome.case == "alias")
    assert alias.success is True
    assert alias.output == 0.25


def test_readme_or_comment_match_never_counts_as_connection(
    readme_only_failure_fixture,
):
    result = {
        item.label: item for item in audit_failure_modes(readme_only_failure_fixture)
    }["structural distortion"]
    assert result.status == "declared_only"
    assert result.aggregation_path is None


def test_aliases_require_an_exact_target_phrase(source_audit_fixture):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    payload = (source.root / path).read_bytes().replace(
        b'"structural distortion"', b'"shape corruption"'
    )
    source.replace_bytes(path, payload)
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    assert result.status == "missing"
    assert result.aliases == ()


def test_source_locations_hash_only_the_declaring_span(source_audit_fixture):
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    location = result.source_locations[0]
    assert location.source_label == "revidgen"
    assert location.path == "eval/5_tasks/common_manipulation.py"
    assert location.start_line == location.end_line == 3
    assert len(location.span_sha256) == 64


def test_unused_named_parser_and_aggregator_are_not_operationalized(
    source_audit_fixture,
):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    payload = (source.root / path).read_bytes().split(b"def evaluate(payloads):", 1)[0]
    source.replace_bytes(path, payload)
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    assert result.status == "declared_only"
    assert result.fixtures == ()


def test_metric_declarations_must_feed_selected_aggregation(source_audit_fixture):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    payload = (source.root / path).read_bytes().replace(
        b"return {METRIC_IDENTIFIER: aggregate_structural_distortion(scores), \"dimensions\": DIMENSIONS}",
        b"return {\"unrelated\": aggregate_structural_distortion(scores)}",
    )
    source.replace_bytes(path, payload)
    traces = trace_metrics(source_audit_fixture)
    assert {trace.partition for trace in traces} == {"embodiment"}


def test_fixture_range_is_recovered_from_parser_ast(source_audit_fixture):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    payload = (source.root / path).read_bytes().replace(b"min(1.0,", b"min(0.25,")
    source.replace_bytes(path, payload)
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    outcomes = {outcome.case: outcome for outcome in result.fixtures}
    assert result.output_range == [0.0, 0.25]
    assert outcomes["boundary"].output == 0.25


def test_discarded_aggregate_output_is_not_operationalized(source_audit_fixture):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    payload = (source.root / path).read_bytes().replace(
        b'return {METRIC_IDENTIFIER: aggregate_structural_distortion(scores), "dimensions": DIMENSIONS}',
        b'aggregate_structural_distortion(scores)\n    return {METRIC_IDENTIFIER: None, "dimensions": DIMENSIONS}',
    )
    source.replace_bytes(path, payload)
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    assert result.status == "declared_only"


def test_alias_fixture_reports_unavailable_without_distinct_alias(source_audit_fixture):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    payload = (source.root / path).read_bytes().replace(
        b'"structural_distortion"', b'"structural distortion"'
    )
    source.replace_bytes(path, payload)
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    alias = next(outcome for outcome in result.fixtures if outcome.case == "alias")
    assert alias.success is False
    assert alias.error == "distinct alias unavailable"


def test_parser_with_unrecovered_control_flow_is_not_operationalized(
    source_audit_fixture,
):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    payload = (source.root / path).read_bytes().replace(
        b"    return min(1.0, max(0.0, float(value)))",
        b"    if value == 'skip':\n        return 0.75\n    return min(1.0, max(0.0, float(value)))",
    )
    source.replace_bytes(path, payload)
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    assert result.status == "declared_only"


def test_shell_entry_point_cannot_connect_another_source(tmp_path, source_audit_fixture):
    evaluator_path = "eval/5_tasks/common_manipulation.py"
    evaluator = (source_audit_fixture["revidgen"].root / evaluator_path).read_bytes()
    sources = {
        "launcher": acquired_source(
            tmp_path,
            "revidgen",
            {"scripts/rbench_eval_5tasks.sh": f"python {evaluator_path}\n".encode()},
        ),
        "evaluator": acquired_source(
            tmp_path,
            "rbench-leaderboard-paper-era",
            {evaluator_path: evaluator},
        ),
    }
    result = {item.label: item for item in audit_failure_modes(sources)}[
        "structural distortion"
    ]
    assert result.status == "declared_only"


def test_docstring_phrase_cannot_unlock_operational_route(source_audit_fixture):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    payload = (source.root / path).read_bytes().replace(
        b'FAILURE_MODES = {"structural_distortion": "structural distortion"}',
        b'"""structural distortion"""',
    )
    source.replace_bytes(path, payload)
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    assert result.status == "declared_only"


def test_reassigned_parser_output_is_not_connected(source_audit_fixture):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    payload = (source.root / path).read_bytes().replace(
        b"    return {METRIC_IDENTIFIER: aggregate_structural_distortion(scores),",
        b"    scores = [0.0]\n    return {METRIC_IDENTIFIER: aggregate_structural_distortion(scores),",
    )
    source.replace_bytes(path, payload)
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    assert result.status == "declared_only"
    assert {trace.partition for trace in trace_metrics(source_audit_fixture)} == {
        "embodiment"
    }


def test_metric_declarations_must_appear_in_returned_structure(source_audit_fixture):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    payload = (source.root / path).read_bytes().replace(
        b'return {METRIC_IDENTIFIER: aggregate_structural_distortion(scores), "dimensions": DIMENSIONS}',
        b'print(METRIC_IDENTIFIER, DIMENSIONS)\n    return aggregate_structural_distortion(scores)',
    )
    source.replace_bytes(path, payload)
    assert {trace.partition for trace in trace_metrics(source_audit_fixture)} == {
        "embodiment"
    }


def test_echoed_evaluator_path_is_not_an_entry_point(source_audit_fixture):
    source_audit_fixture["revidgen"].replace_bytes(
        "scripts/rbench_eval_5tasks.sh",
        b"echo eval/5_tasks/common_manipulation.py\n",
    )
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    assert result.status == "declared_only"


def test_boundary_fixture_uses_recovered_upper_bound(source_audit_fixture):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    payload = (source.root / path).read_bytes().replace(
        b"min(1.0, max(0.0,", b"min(10.0, max(5.0,"
    )
    source.replace_bytes(path, payload)
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    boundary = next(outcome for outcome in result.fixtures if outcome.case == "boundary")
    assert result.output_range == [5.0, 10.0]
    assert boundary.output == 10.0


def test_evaluator_path_used_as_python_config_is_not_invoked(source_audit_fixture):
    source_audit_fixture["revidgen"].replace_bytes(
        "scripts/rbench_eval_5tasks.sh",
        b"python runner.py --config eval/5_tasks/common_manipulation.py\n",
    )
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    assert result.status == "declared_only"


def test_aggregate_must_be_value_of_metric_identifier(source_audit_fixture):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    payload = (source.root / path).read_bytes().replace(
        b'{METRIC_IDENTIFIER: aggregate_structural_distortion(scores), "dimensions": DIMENSIONS}',
        b'{"other": aggregate_structural_distortion(scores), METRIC_IDENTIFIER: None, "dimensions": DIMENSIONS}',
    )
    source.replace_bytes(path, payload)
    assert {trace.partition for trace in trace_metrics(source_audit_fixture)} == {
        "embodiment"
    }


def test_echoed_python_command_is_not_an_entry_point(source_audit_fixture):
    source_audit_fixture["revidgen"].replace_bytes(
        "scripts/rbench_eval_5tasks.sh",
        b"echo python eval/5_tasks/common_manipulation.py\n",
    )
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    assert result.status == "declared_only"


def test_expression_that_discards_parser_result_is_not_connected(source_audit_fixture):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    payload = (source.root / path).read_bytes().replace(
        b"scores = [parse_structural_distortion(payload) for payload in payloads]",
        b"scores = (parse_structural_distortion(payloads[0]), [0.0])[1]",
    )
    source.replace_bytes(path, payload)
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    assert result.status == "declared_only"


def test_aggregate_argument_must_be_parser_derived_value(source_audit_fixture):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    payload = (source.root / path).read_bytes().replace(
        b"aggregate_structural_distortion(scores)",
        b"aggregate_structural_distortion((scores, [0.0])[1])",
    )
    source.replace_bytes(path, payload)
    result = {
        item.label: item for item in audit_failure_modes(source_audit_fixture)
    }["structural distortion"]
    assert result.status == "declared_only"


def test_dimensions_must_use_dimensions_result_field(source_audit_fixture):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    payload = (source.root / path).read_bytes().replace(
        b'"dimensions": DIMENSIONS', b'"debug": DIMENSIONS'
    )
    source.replace_bytes(path, payload)
    assert {trace.partition for trace in trace_metrics(source_audit_fixture)} == {
        "embodiment"
    }


def test_text_location_hashes_original_invalid_utf8_bytes(
    readme_only_failure_fixture,
):
    payload = b"\xff structural distortion\n"
    readme_only_failure_fixture["revidgen"].replace_bytes("README.md", payload)
    result = {
        item.label: item for item in audit_failure_modes(readme_only_failure_fixture)
    }["structural distortion"]
    readme_location = next(
        location for location in result.source_locations if location.path == "README.md"
    )
    assert readme_location.span_sha256 == sha256(payload).hexdigest()


def test_metric_provenance_includes_dimensions_declaration(source_audit_fixture):
    task = next(trace for trace in trace_metrics(source_audit_fixture) if trace.partition == "task")
    assert any(location.start_line == 2 for location in task.source_locations)


def test_ast_location_hashes_original_declared_encoding_bytes(source_audit_fixture):
    source = source_audit_fixture["revidgen"]
    path = "eval/5_tasks/common_manipulation.py"
    original = (source.root / path).read_bytes()
    first_line, remainder = original.split(b"\n", 1)
    payload = b"# coding: latin-1\n" + first_line + b"  # \xe9\n" + remainder
    source.replace_bytes(path, payload)
    task = next(trace for trace in trace_metrics(source_audit_fixture) if trace.partition == "task")
    identifier = next(location for location in task.source_locations if location.start_line == 2)
    expected_line = payload.splitlines(keepends=True)[1]
    assert identifier.span_sha256 == sha256(expected_line).hexdigest()
