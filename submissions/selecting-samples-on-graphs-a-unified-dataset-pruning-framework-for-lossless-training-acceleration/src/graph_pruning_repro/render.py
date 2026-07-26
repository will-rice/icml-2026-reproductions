"""Deterministic report and poster rendering from accepted evidence only."""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


_REQUIRED_TOP_LEVEL = {
    "schema_version",
    "attempt_id",
    "source_revision",
    "paper",
    "target_claims",
    "environment",
    "transcriptions",
    "searches",
    "witnesses",
    "guarantee_violations",
    "out_of_premise_diagnostics",
    "proof_ledger",
    "claim_results",
    "unavailable_claims",
    "commands",
    "artifacts",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_accepted_evidence() -> dict[str, object]:
    """Load the canonical evidence that Task 6 already accepted."""

    path = _project_root() / "evidence" / "evidence.json"
    evidence = json.loads(path.read_text())
    if type(evidence) is not dict or set(evidence) != _REQUIRED_TOP_LEVEL:
        raise ValueError("accepted evidence top-level schema drift")
    return evidence


def resolve_rfc6901(evidence: object, pointer: str) -> object:
    """Resolve one canonical RFC 6901 pointer with numeric array indices."""

    if type(pointer) is not str or not pointer.startswith("/"):
        raise ValueError("evidence pointer must start with /")
    current = evidence
    for encoded in pointer[1:].split("/"):
        if "{" in encoded or "}" in encoded:
            raise ValueError("ID-as-index pseudo-pointers are forbidden")
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if re.fullmatch(r"0|[1-9][0-9]*", token) is None:
                raise ValueError("array evidence pointer must use a numeric index")
            index = int(token)
            if index >= len(current):
                raise ValueError("array evidence pointer is out of range")
            current = current[index]
        elif isinstance(current, Mapping):
            if token not in current:
                raise ValueError("object evidence pointer is missing")
            current = current[token]
        else:
            raise ValueError("evidence pointer traverses a scalar")
    return current


def _index_by(
    records: Sequence[Mapping[str, object]],
    field: str,
    value: object,
) -> int:
    matches = [
        index
        for index, record in enumerate(records)
        if record.get(field) == value
    ]
    if len(matches) != 1:
        raise ValueError(f"accepted evidence lacks unique {field}={value!r}")
    return matches[0]


def _leaf_pointers(value: object, prefix: str) -> list[str]:
    pointers: list[str] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            encoded = str(key).replace("~", "~0").replace("/", "~1")
            pointers.extend(_leaf_pointers(value[key], f"{prefix}/{encoded}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            pointers.extend(_leaf_pointers(child, f"{prefix}/{index}"))
    elif value is None or isinstance(value, (str, int, bool)):
        pointers.append(prefix)
    return pointers


def rendered_pointer_values(
    evidence: Mapping[str, object],
) -> tuple[tuple[str, object], ...]:
    """Return every evidence value selected for deterministic display."""

    if set(evidence) != _REQUIRED_TOP_LEVEL:
        raise ValueError("accepted evidence top-level schema drift")
    pointers = [
        "/target_claims/0",
        "/target_claims/1",
        "/paper/revision",
        "/paper/source_url",
        "/paper/pdf_byte_count",
        "/paper/pdf_sha256",
        "/paper/license",
        "/source_revision",
        "/environment/compute",
        "/environment/network_used",
        "/environment/paid_api_cost_usd",
        "/commands/0/actual",
        "/commands/0/ceiling",
    ]

    for index, record in enumerate(evidence["transcriptions"]["records"]):
        pointers.extend(
            (
                f"/transcriptions/records/{index}/equation",
                f"/transcriptions/records/{index}/section",
                f"/transcriptions/records/{index}/normalized_expression",
            )
        )

    for search_index, search in enumerate(evidence["searches"]):
        pointers.extend(
            (
                f"/searches/{search_index}/id",
                f"/searches/{search_index}/evidence_kind",
                f"/searches/{search_index}/completed",
            )
        )
        for component_index, _ in enumerate(search["components"]):
            pointers.extend(
                (
                    f"/searches/{search_index}/components/{component_index}/id",
                    (
                        f"/searches/{search_index}/components/"
                        f"{component_index}/actual"
                    ),
                    (
                        f"/searches/{search_index}/components/"
                        f"{component_index}/declared_ceiling"
                    ),
                )
            )

    for witness_index, witness in enumerate(evidence["witnesses"]):
        pointers.extend(_leaf_pointers(witness, f"/witnesses/{witness_index}"))

    for result_index, result in enumerate(evidence["claim_results"]):
        pointers.extend(
            (
                f"/claim_results/{result_index}/audit",
                f"/claim_results/{result_index}/model_variant",
                f"/claim_results/{result_index}/evidence_kind",
                f"/claim_results/{result_index}/status",
            )
        )

    for unavailable_index, unavailable in enumerate(
        evidence["unavailable_claims"]
    ):
        pointers.extend(
            (
                f"/unavailable_claims/{unavailable_index}/id",
                f"/unavailable_claims/{unavailable_index}/status",
                f"/unavailable_claims/{unavailable_index}/reason",
            )
        )

    diagnostics = evidence["out_of_premise_diagnostics"]
    ratio_pointer: str | None = None
    for index, diagnostic in enumerate(diagnostics):
        ratio = diagnostic["ratio_classification"].get("ratio")
        if ratio is not None:
            pointers.append(f"/out_of_premise_diagnostics/{index}/id")
            ratio_pointer = (
                f"/out_of_premise_diagnostics/{index}/"
                "ratio_classification/ratio"
            )
            pointers.append(
                f"/out_of_premise_diagnostics/{index}/"
                "ratio_classification/status"
            )
            break
    if ratio_pointer is not None:
        pointers.append(ratio_pointer)

    symbolic = evidence["proof_ledger"]["symbolic"]["ledgers"]
    for model_variant in (
        "paper_samplewise_literal",
        "appendix_inline_shift_literal",
        "modular_shift_candidate",
    ):
        rows = symbolic[model_variant]
        variant_token = model_variant.replace("~", "~0").replace("/", "~1")
        for row_index, row in enumerate(rows):
            pointers.append(
                (
                    f"/proof_ledger/symbolic/ledgers/{variant_token}/"
                    f"{row_index}/equation"
                )
            )
            for conclusion_index, _ in enumerate(row["conclusions"]):
                pointers.extend(
                    (
                        (
                            f"/proof_ledger/symbolic/ledgers/{variant_token}/"
                            f"{row_index}/conclusions/{conclusion_index}/check_id"
                        ),
                        (
                            f"/proof_ledger/symbolic/ledgers/{variant_token}/"
                            f"{row_index}/conclusions/{conclusion_index}/status"
                        ),
                    )
                )

    unique: list[str] = []
    seen: set[str] = set()
    for pointer in pointers:
        if pointer not in seen:
            resolve_rfc6901(evidence, pointer)
            seen.add(pointer)
            unique.append(pointer)
    return tuple((pointer, resolve_rfc6901(evidence, pointer)) for pointer in unique)


def render_distribution_assets(
    evidence: Mapping[str, object],
) -> dict[str, str]:
    """Return the exact attribution and license assets for distribution."""

    if evidence["paper"]["license"] != "CC BY-NC-SA 4.0":
        raise ValueError("accepted evidence paper-license boundary drift")
    root = _project_root()
    paths = (
        "NOTICE.md",
        "LICENSE",
        "LICENSES/CC-BY-NC-SA-4.0.txt",
    )
    return {path: (root / path).read_text() for path in paths}


def _report_value(
    label: str,
    pointer: str,
    evidence: Mapping[str, object],
) -> str:
    value = resolve_rfc6901(evidence, pointer)
    return f"- {label}: {value} _(evidence: `{pointer}`)_"


def _poster_value(
    label: str,
    pointer: str,
    evidence: Mapping[str, object],
) -> str:
    value = html.escape(str(resolve_rfc6901(evidence, pointer)))
    return (
        f"<li><strong>{html.escape(label)}:</strong> "
        f'<span data-evidence-path="{html.escape(pointer)}">{value}</span></li>'
    )


def _appendix_witness_index(evidence: Mapping[str, object]) -> int:
    return _index_by(
        evidence["witnesses"],
        "property",
        "appendix_inline_shift_diminishing_returns",
    )


def _cardinality_witness_index(evidence: Mapping[str, object]) -> int:
    return _index_by(
        evidence["witnesses"],
        "property",
        "optimum_remainder_cardinality_exceeds_b_minus_t",
    )


def _first_defined_ratio_index(evidence: Mapping[str, object]) -> int:
    for index, diagnostic in enumerate(
        evidence["out_of_premise_diagnostics"]
    ):
        if diagnostic["ratio_classification"].get("ratio") is not None:
            return index
    raise ValueError("accepted evidence lacks an exact ratio diagnostic")


def _objective_witness_index(evidence: Mapping[str, object]) -> int:
    return _index_by(
        evidence["witnesses"],
        "property",
        "paper_mwcp_vs_paper_samplewise_literal",
    )


def render_report(evidence: Mapping[str, object]) -> str:
    """Render a deterministic Markdown report from accepted evidence."""

    appendix_index = _appendix_witness_index(evidence)
    cardinality_index = _cardinality_witness_index(evidence)
    objective_index = _objective_witness_index(evidence)
    ratio_index = _first_defined_ratio_index(evidence)
    lines = [
        "# Graph Dataset Pruning Formal-Evidence Reproduction",
        "",
        "## Target claims and interpretation boundary",
        "",
        _report_value("Target claim", "/target_claims/0", evidence),
        _report_value("Target claim", "/target_claims/1", evidence),
        "",
        (
            "This report recomputes formal evidence from the released paper "
            "artifact. It does not present paper-reported experiments as "
            "reproduced measurements."
        ),
        "",
        "## Literal and repaired formulations",
        "",
        (
            "`paper_mwcp` and `paper_samplewise_literal` remain separate: "
            "the literal samplewise formulation double-counts symmetric "
            "pair interactions. `single_counted_pairwise` and "
            "`half_corrected_samplewise` are repaired comparisons, not "
            "paper implementations."
        ),
        (
            "`appendix_inline_shift_literal` is the Appendix-inline quadratic "
            "shift. `modular_shift_candidate` is a distinct repaired modular "
            "candidate and never inherits the literal witness."
        ),
        _report_value(
            "MWCP witness value",
            f"/witnesses/{objective_index}/comparison/paper_mwcp",
            evidence,
        ),
        _report_value(
            "Literal samplewise witness value",
            (
                f"/witnesses/{objective_index}/comparison/"
                "paper_samplewise_literal"
            ),
            evidence,
        ),
        "",
        "## Appendix shift and proof boundaries",
        "",
        (
            "The literal Appendix witness is the **1 then 3** marginal "
            "sequence "
            f"_(evidence: `/witnesses/{appendix_index}/intermediate_values/"
            "marginal_empty` and "
            f"`/witnesses/{appendix_index}/intermediate_values/marginal_y`)_"
            "; the exact rational values follow."
        ),
        _report_value(
            "First marginal",
            (
                f"/witnesses/{appendix_index}/intermediate_values/"
                "marginal_empty"
            ),
            evidence,
        ),
        _report_value(
            "Second marginal",
            (
                f"/witnesses/{appendix_index}/intermediate_values/"
                "marginal_y"
            ),
            evidence,
        ),
        _report_value(
            "Remainder cardinality",
            (
                f"/witnesses/{cardinality_index}/intermediate_values/"
                "remainder_cardinality"
            ),
            evidence,
        ),
        _report_value(
            "Claimed b-minus-t cardinality",
            (
                f"/witnesses/{cardinality_index}/intermediate_values/"
                "b_minus_t"
            ),
            evidence,
        ),
        (
            "The cardinality record is independent of weights. It contradicts "
            "only the stated b-minus-t step; it is not labeled an Eq. "
            "counterexample or theorem counterexample."
        ),
        "",
        "## Greedy, optimum, and ratio evidence",
        "",
        (
            "The accepted aggregate retains exact ratio classifications and "
            "the independently recomputed greedy/optimum accounting. It does "
            "not retain terminal greedy and optimum values as displayable "
            "records, so this renderer does not invent them."
        ),
        _report_value(
            "Representative diagnostic",
            f"/out_of_premise_diagnostics/{ratio_index}/id",
            evidence,
        ),
        _report_value(
            "Ratio classification",
            (
                f"/out_of_premise_diagnostics/{ratio_index}/"
                "ratio_classification/status"
            ),
            evidence,
        ),
        _report_value(
            "Exact ratio",
            (
                f"/out_of_premise_diagnostics/{ratio_index}/"
                "ratio_classification/ratio"
            ),
            evidence,
        ),
        (
            "This ratio belongs to an explicitly out-of-premise diagnostic; "
            "it is not presented as a guarantee violation."
        ),
        "",
        "## Independent audit classifications",
        "",
    ]
    for index, result in enumerate(evidence["claim_results"]):
        lines.extend(
            (
                _report_value(
                    "Audit",
                    f"/claim_results/{index}/audit",
                    evidence,
                ),
                _report_value(
                    "Model variant",
                    f"/claim_results/{index}/model_variant",
                    evidence,
                ),
                _report_value(
                    "Classification",
                    f"/claim_results/{index}/status",
                    evidence,
                ),
                "",
            )
        )

    lines.extend(
        (
            "## Exhaustive domains and ceilings",
            "",
            (
                "Every actual below is paired with its declared ceiling. "
                "A ceiling is a limit, not an equality target."
            ),
        )
    )
    for search_index, search in enumerate(evidence["searches"]):
        lines.append(
            _report_value(
                "Audit group",
                f"/searches/{search_index}/id",
                evidence,
            )
        )
        for component_index, _ in enumerate(search["components"]):
            lines.extend(
                (
                    _report_value(
                        "Component",
                        (
                            f"/searches/{search_index}/components/"
                            f"{component_index}/id"
                        ),
                        evidence,
                    ),
                    _report_value(
                        "Actual",
                        (
                            f"/searches/{search_index}/components/"
                            f"{component_index}/actual"
                        ),
                        evidence,
                    ),
                    _report_value(
                        "Declared ceiling",
                        (
                            f"/searches/{search_index}/components/"
                            f"{component_index}/declared_ceiling"
                        ),
                        evidence,
                    ),
                )
            )
        lines.append("")

    lines.extend(
        (
            "## Appendix proof-ledger rows",
            "",
            (
                "The normalized nested conclusion records keep the literal, "
                "Appendix-inline, and modular candidate proof paths separate."
            ),
        )
    )
    symbolic = evidence["proof_ledger"]["symbolic"]["ledgers"]
    for model_variant in (
        "paper_samplewise_literal",
        "appendix_inline_shift_literal",
        "modular_shift_candidate",
    ):
        lines.append(f"### `{model_variant}`")
        lines.append("")
        token = model_variant.replace("~", "~0").replace("/", "~1")
        for row_index, row in enumerate(symbolic[model_variant]):
            lines.append(
                _report_value(
                    "Equation",
                    (
                        f"/proof_ledger/symbolic/ledgers/{token}/"
                        f"{row_index}/equation"
                    ),
                    evidence,
                )
            )
            for conclusion_index, _ in enumerate(row["conclusions"]):
                lines.extend(
                    (
                        _report_value(
                            "Check",
                            (
                                f"/proof_ledger/symbolic/ledgers/{token}/"
                                f"{row_index}/conclusions/{conclusion_index}/"
                                "check_id"
                            ),
                            evidence,
                        ),
                        _report_value(
                            "Status",
                            (
                                f"/proof_ledger/symbolic/ledgers/{token}/"
                                f"{row_index}/conclusions/{conclusion_index}/"
                                "status"
                            ),
                            evidence,
                        ),
                    )
                )
        lines.append("")

    lines.extend(
        (
            "## Provenance and reviewed equations",
            "",
            _report_value("Pinned revision", "/paper/revision", evidence),
            _report_value("Source", "/paper/source_url", evidence),
            _report_value("PDF bytes", "/paper/pdf_byte_count", evidence),
            _report_value("PDF digest", "/paper/pdf_sha256", evidence),
        )
    )
    for index, _ in enumerate(evidence["transcriptions"]["records"]):
        lines.extend(
            (
                _report_value(
                    "Equation",
                    f"/transcriptions/records/{index}/equation",
                    evidence,
                ),
                _report_value(
                    "Reviewed expression",
                    (
                        f"/transcriptions/records/{index}/"
                        "normalized_expression"
                    ),
                    evidence,
                ),
            )
        )

    lines.extend(
        (
            "",
            "## Limitations",
            "",
            (
                "Bounded enumeration can refute but cannot prove "
                "arbitrary-real universal claims. No released implementation "
                "resolves the paper's edge-counting ambiguity or Appendix "
                "shift ambiguity."
            ),
            "",
            "## Unavailable empirical claims",
            "",
        )
    )
    for index, _ in enumerate(evidence["unavailable_claims"]):
        lines.extend(
            (
                _report_value(
                    "Status",
                    f"/unavailable_claims/{index}/status",
                    evidence,
                ),
                _report_value(
                    "Boundary",
                    f"/unavailable_claims/{index}/reason",
                    evidence,
                ),
            )
        )

    lines.extend(
        (
            "",
            "## Attribution and licenses",
            "",
            (
                "Seven-author paper attribution and adaptation details are "
                "in `NOTICE.md`. Original executable code and schema are "
                "covered by `LICENSE`; transcriptions and evidence are "
                "covered by `LICENSES/CC-BY-NC-SA-4.0.txt`."
            ),
            _report_value("Paper asset license", "/paper/license", evidence),
            "",
            "## Complete display-pointer ledger",
            "",
        )
    )
    for pointer, _ in rendered_pointer_values(evidence):
        lines.append(_report_value("Displayed evidence", pointer, evidence))
    return "\n".join(lines) + "\n"


def render_poster(evidence: Mapping[str, object]) -> str:
    """Render a deterministic self-contained evidence poster."""

    appendix_index = _appendix_witness_index(evidence)
    ratio_index = _first_defined_ratio_index(evidence)
    body = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Graph Dataset Pruning Formal Evidence</title>",
        "</head>",
        "<body>",
        "<header>",
        "<h1>Graph Dataset Pruning Formal-Evidence Reproduction</h1>",
        "<p>Evidence-only theorem audit; no training measurements.</p>",
        "</header>",
        "<main>",
        "<section>",
        "<h2>Target claims</h2>",
        "<ul>",
        _poster_value("Claim", "/target_claims/0", evidence),
        _poster_value("Claim", "/target_claims/1", evidence),
        "</ul>",
        "</section>",
        "<section>",
        "<h2>Literal and repaired boundaries</h2>",
        (
            "<p><code>paper_samplewise_literal</code> remains distinct from "
            "<code>paper_mwcp</code>, <code>single_counted_pairwise</code>, "
            "and <code>half_corrected_samplewise</code>.</p>"
        ),
        (
            "<p><code>appendix_inline_shift_literal</code> is not "
            "<code>modular_shift_candidate</code>.</p>"
        ),
        (
            f"<p>The literal Appendix witness is "
            f'<span data-evidence-path="/witnesses/{appendix_index}/'
            'intermediate_values/marginal_empty">1</span> then'
        ),
        (
            f' <span data-evidence-path="/witnesses/{appendix_index}/'
            'intermediate_values/marginal_y">3</span>; exact rationals are '
            "listed in the pointer ledger.</p>"
        ),
        "</section>",
        "<section>",
        "<h2>Formal-evidence boundary</h2>",
        (
            "<p>Exact ratio classifications and independently recomputed "
            "greedy/optimum accounting are displayed from evidence. Terminal "
            "greedy and optimum values are not retained as display records "
            "and are not invented here.</p>"
        ),
        "<ul>",
        _poster_value(
            "Ratio classification",
            (
                f"/out_of_premise_diagnostics/{ratio_index}/"
                "ratio_classification/status"
            ),
            evidence,
        ),
        _poster_value(
            "Exact out-of-premise ratio",
            (
                f"/out_of_premise_diagnostics/{ratio_index}/"
                "ratio_classification/ratio"
            ),
            evidence,
        ),
        "</ul>",
        (
            "<p>Bounded enumeration can refute but cannot prove "
            "arbitrary-real universal claims. No released implementation "
            "resolves edge counting or the shift.</p>"
        ),
        "</section>",
        "<section>",
        "<h2>Unavailable empirical claims</h2>",
        "<ul>",
    ]
    for index, _ in enumerate(evidence["unavailable_claims"]):
        body.extend(
            (
                _poster_value(
                    "Status",
                    f"/unavailable_claims/{index}/status",
                    evidence,
                ),
                _poster_value(
                    "Boundary",
                    f"/unavailable_claims/{index}/reason",
                    evidence,
                ),
            )
        )
    body.extend(
        (
            "</ul>",
            "</section>",
            "<section>",
            "<h2>Provenance, attribution, and licenses</h2>",
            "<ul>",
            _poster_value("Source", "/paper/source_url", evidence),
            _poster_value("Revision", "/paper/revision", evidence),
            _poster_value("Paper asset license", "/paper/license", evidence),
            "</ul>",
            (
                "<p>Seven-author attribution: <code>NOTICE.md</code>. "
                "Original code and schema: <code>LICENSE</code>. "
                "Transcriptions and evidence: "
                "<code>LICENSES/CC-BY-NC-SA-4.0.txt</code>.</p>"
            ),
            "</section>",
            "<details>",
            "<summary>Complete display-pointer ledger</summary>",
            "<ul>",
        )
    )
    for pointer, _ in rendered_pointer_values(evidence):
        body.append(_poster_value("Displayed evidence", pointer, evidence))
    body.extend(
        (
            "</ul>",
            "</details>",
            "</main>",
            "</body>",
            "</html>",
            "",
        )
    )
    return "\n".join(body)


def assert_render_agreement(
    evidence: Mapping[str, object],
    report: str,
    poster: str,
) -> None:
    """Reject any selected display value lacking its canonical pointer."""

    if type(report) is not str or type(poster) is not str:
        raise TypeError("rendered report and poster must be strings")
    for pointer, value in rendered_pointer_values(evidence):
        if "{id}" in pointer:
            raise ValueError("rendered evidence contains an ID pseudo-pointer")
        if resolve_rfc6901(evidence, pointer) != value:
            raise ValueError("rendered evidence pointer resolution drift")
        if f"evidence: `{pointer}`" not in report or str(value) not in report:
            raise ValueError("report value lacks canonical evidence metadata")
        marker = f'data-evidence-path="{pointer}"'
        if marker not in poster or html.escape(str(value)) not in poster:
            raise ValueError("poster value lacks canonical evidence metadata")
