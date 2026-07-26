"""Static HTML and README rendering for PostTrainBench reproduction.

All rendered values trace to a canonical JSON pointer.  No manually
copied result value appears in the output.  Templates are plain Python
string formatting to avoid runtime template engine variation.
"""

from __future__ import annotations

from typing import Any

from posttrainbench_repro.constants import MODEL_ORDER


# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg: #0f172a;
  --surface: #1e293b;
  --border: #334155;
  --text: #e2e8f0;
  --text-muted: #94a3b8;
  --accent: #38bdf8;
  --green: #4ade80;
  --yellow: #facc15;
  --red: #f87171;
  --orange: #fb923c;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}
h1 { font-size: 1.75rem; margin-bottom: 0.5rem; color: var(--accent); }
h2 { font-size: 1.25rem; margin: 1.5rem 0 0.75rem; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 0.25rem; }
h3 { font-size: 1rem; margin: 1rem 0 0.5rem; color: var(--text); }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0; }
th, td { padding: 0.4rem 0.75rem; text-align: left; border: 1px solid var(--border); font-size: 0.85rem; }
th { background: var(--surface); color: var(--accent); font-weight: 600; }
td { background: var(--bg); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; margin: 0.75rem 0; }
.status { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
.status-partial { background: rgba(250, 204, 21, 0.15); color: var(--yellow); border: 1px solid var(--yellow); }
.status-unavailable { background: rgba(248, 113, 113, 0.15); color: var(--red); border: 1px solid var(--red); }
.status-verified { background: rgba(74, 222, 128, 0.15); color: var(--green); border: 1px solid var(--green); }
.ptr { color: var(--text-muted); font-family: monospace; font-size: 0.75rem; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
ul, ol { margin: 0.5rem 0 0.5rem 1.5rem; }
li { margin: 0.25rem 0; font-size: 0.9rem; }
.limitation { background: rgba(251, 146, 60, 0.1); border-left: 3px solid var(--orange); padding: 0.5rem 0.75rem; margin: 0.25rem 0; font-size: 0.85rem; }
.excerpt { background: var(--bg); border: 1px solid var(--border); padding: 0.5rem; margin: 0.25rem 0; font-family: monospace; font-size: 0.8rem; white-space: pre-wrap; }
nav { margin-bottom: 1.5rem; }
nav a { margin-right: 1rem; }
footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.8rem; color: var(--text-muted); }
"""


def _status_span(status: str) -> str:
    """Render a status badge."""
    cls = {
        "partial-support": "status-partial",
        "unavailable": "status-unavailable",
        "verified": "status-verified",
    }.get(status, "status-partial")
    return f'<span class="status {cls}">{status}</span>'


def _ptr(pointer: str) -> str:
    """Render a JSON pointer reference."""
    return f'<span class="ptr">{pointer}</span>'


def _evidence_value(value_html: str, pointer: str) -> str:
    """Bind one visible value to the exact canonical JSON value it renders."""
    return (
        '<span class="evidence-result" '
        f'data-evidence-pointer="{_escape(pointer)}">'
        f"{value_html} {_ptr(pointer)}</span>"
    )


def _status_value(status: str, pointer: str) -> str:
    return _evidence_value(_status_span(status), pointer)


def _escape(text: str) -> str:
    """HTML-escape text."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Coverage matrix HTML
# ---------------------------------------------------------------------------

def _render_matrix(coverage: dict[str, Any]) -> str:
    """Render the 4×7 coverage matrix table."""
    cell_counts = coverage.get("cell_counts", {})
    benchmarks = sorted(cell_counts.keys())

    rows = []
    rows.append("<table>")
    rows.append("<tr><th>Benchmark</th>")
    for model in MODEL_ORDER:
        rows.append(f"<th>{_escape(model)}</th>")
    rows.append("</tr>")

    for bench in benchmarks:
        counts = cell_counts[bench]
        rows.append(f"<tr><td><code>{bench}</code></td>")
        for model_index, count in enumerate(counts):
            pointer = (
                f"evidence/coverage.json#/cell_counts/{bench}/{model_index}"
            )
            rows.append(
                f"<td>{_evidence_value(str(count), pointer)}</td>"
            )
        rows.append("</tr>")

    rows.append("</table>")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Reward-hacking modes HTML
# ---------------------------------------------------------------------------

def _render_reward_hacking(rh: dict[str, Any]) -> str:
    """Render the three reward-hacking submodes."""
    parts: list[str] = []

    # Mode 1: Contamination
    contam = rh["training_on_test_sets"]
    contam_base = "evidence/reward_hacking.json#/training_on_test_sets"
    parts.append('<div class="card">')
    parts.append(
        "<h3>Training on Test Sets "
        f'{_status_value(contam["status"], contam_base + "/status")}</h3>'
    )
    parts.append(
        "<p>Witness: "
        + _evidence_value(
            f'<code>{_escape(contam["witness_path"])}</code>',
            contam_base + "/witness_path",
        )
        + "</p>"
    )
    parts.append(
        "<p>SHA-256: "
        + _evidence_value(
            f'<code>{contam["witness_sha256"]}</code>',
            contam_base + "/witness_sha256",
        )
        + "</p>"
    )
    parts.append(
        "<p>Content: "
        + _evidence_value(
            f'<code>{_escape(contam["witness_bytes"])}</code>',
            contam_base + "/witness_bytes",
        )
        + "</p>"
    )
    parts.append(
        "<p>Type: "
        + _evidence_value(
            _escape(contam["observation_type"]),
            contam_base + "/observation_type",
        )
        + "</p>"
    )
    parts.append("</div>")

    # Mode 2: Instruction-model
    instr = rh["downloading_instruction_tuned_checkpoint"]
    instr_base = (
        "evidence/reward_hacking.json#/"
        "downloading_instruction_tuned_checkpoint"
    )
    parts.append('<div class="card">')
    parts.append(
        "<h3>Instruction-Tuned Checkpoint "
        f'{_status_value(instr["status"], instr_base + "/status")}</h3>'
    )
    parts.append(
        "<p>Judgment: "
        + _evidence_value(
            f'<code>{_escape(instr["judgment_path"])}</code>',
            instr_base + "/judgment_path",
        )
        + "</p>"
    )
    parts.append(
        "<p>Judgment SHA-256: "
        + _evidence_value(
            f'<code>{instr["judgment_sha256"]}</code>',
            instr_base + "/judgment_sha256",
        )
        + "</p>"
    )
    parts.append(
        "<p>Content: "
        + _evidence_value(
            f'<code>{_escape(instr["judgment_bytes"])}</code>',
            instr_base + "/judgment_bytes",
        )
        + " (no trailing newline)</p>"
    )
    parts.append("<h4>Safe Trace Excerpts</h4>")
    for index, excerpt in enumerate(instr.get("safe_excerpts", [])):
        excerpt_base = f"{instr_base}/safe_excerpts/{index}"
        parts.append(
            '<div class="excerpt">Record '
            + _evidence_value(
                str(excerpt["record"]),
                excerpt_base + "/record",
            )
            + " "
            + _evidence_value(
                _escape(excerpt["json_pointer"]),
                excerpt_base + "/json_pointer",
            )
            + ": "
            + _evidence_value(
                _escape(str(excerpt["text"])),
                excerpt_base + "/text",
            )
            + "\nSHA-256: "
            + _evidence_value(
                excerpt["sha256"],
                excerpt_base + "/sha256",
            )
            + "</div>"
        )
    parts.append(
        "<p>Complete trace ("
        + _evidence_value(
            f'{instr["trace"]["size"]:,}',
            instr_base + "/trace/size",
        )
        + " bytes) is "
        + _evidence_value(
            "NOT redistributed",
            instr_base + "/trace/redistributed",
        )
        + ".</p>"
    )
    parts.append("</div>")

    # Mode 3: API misuse
    api = rh["using_discovered_api_key"]
    api_base = "evidence/reward_hacking.json#/using_discovered_api_key"
    parts.append('<div class="card">')
    parts.append(
        "<h3>API Key Discovery "
        f'{_status_value(api["status"], api_base + "/status")}</h3>'
    )
    parts.append(
        "<p>"
        + _evidence_value(
            _escape(api["unavailability_reason"]),
            api_base + "/unavailability_reason",
        )
        + "</p>"
    )
    parts.append("</div>")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Limitations HTML
# ---------------------------------------------------------------------------

def _render_limitations(claims: dict[str, Any]) -> str:
    """Render all limitations from both claims."""
    parts: list[str] = []
    seen: set[str] = set()
    for claim_key in ["claim_1", "claim_2"]:
        claim = claims[claim_key]
        for index, lim in enumerate(claim.get("limitations", [])):
            if lim not in seen:
                seen.add(lim)
                pointer = (
                    f"evidence/claims.json#/{claim_key}/limitations/{index}"
                )
                parts.append(
                    '<div class="limitation">'
                    f"{_evidence_value(_escape(lim), pointer)}</div>"
                )
    return "\n".join(parts)


def _render_claim(claim: dict[str, Any], claim_key: str, title: str) -> str:
    base = f"evidence/claims.json#/{claim_key}"
    return f"""\
<div class="card">
<h3>{title} {_status_value(claim["status"], base + "/status")}</h3>
<p>{_evidence_value(_escape(claim["text"]), base + "/text")}</p>
<p>SHA-256: {_evidence_value(f'<code>{claim["sha256"]}</code>', base + "/sha256")}</p>
<p>{_evidence_value(_escape(claim["summary"]), base + "/summary")}</p>
</div>"""


def _render_coverage_summary(coverage: dict[str, Any]) -> str:
    auxiliary = coverage["excluded_auxiliary_data"]
    values = [
        (
            "Accepted benchmarks",
            coverage["accepted_benchmark_count"],
            "accepted_benchmark_count",
        ),
        (
            "Accepted models",
            coverage["accepted_model_count"],
            "accepted_model_count",
        ),
        ("Recognized tasks", coverage["recognized_task_count"], "recognized_task_count"),
        ("Recognized roots", coverage["recognized_root_count"], "recognized_root_count"),
        (
            "Root/cell pairs",
            coverage["recognized_root_cell_pairs"],
            "recognized_root_cell_pairs",
        ),
        ("Duplicate pairs", coverage["duplicate_job_pairs"], "duplicate_job_pairs"),
        ("Missing pairs", coverage["missing_root_cell_pairs"], "missing_root_cell_pairs"),
        ("Excluded nested task dirs", coverage["excluded_dirs_count"], "excluded_dirs_count"),
    ]
    parts = ['<div class="card"><ul>']
    for label, value, field in values:
        pointer = f"evidence/coverage.json#/{field}"
        parts.append(
            f"<li>{label}: {_evidence_value(str(value), pointer)}</li>"
        )
    parts.append(
        "<li>Excluded auxiliary top-level: "
        + _evidence_value(
            f'<code>{_escape(auxiliary["top_level_path"])}</code>',
            "evidence/coverage.json#/excluded_auxiliary_data/top_level_path",
        )
        + "; files: "
        + _evidence_value(
            str(auxiliary["file_count"]),
            "evidence/coverage.json#/excluded_auxiliary_data/file_count",
        )
        + "; directories: "
        + _evidence_value(
            str(auxiliary["directory_count"]),
            "evidence/coverage.json#/excluded_auxiliary_data/directory_count",
        )
        + "; counted as task root: "
        + _evidence_value(
            str(auxiliary["counted_as_task_root"]).lower(),
            "evidence/coverage.json#/excluded_auxiliary_data/counted_as_task_root",
        )
        + "</li>"
    )
    parts.append("</ul></div>")
    return "\n".join(parts)


def _source_reference_text(reference: dict[str, Any]) -> str:
    if "path" in reference:
        line_text = ",".join(str(line) for line in reference["lines"])
        return (
            f'{reference["commit"]}/{reference["path"]}:line {line_text} '
            f'(blob {reference["git_object_sha1"]}, '
            f'raw SHA-256 {reference["raw_sha256"]})'
        )
    return (
        f'{reference["kind"]}: '
        + ", ".join(reference["paths"])
    )


def _render_protocol(coverage: dict[str, Any]) -> str:
    protocol = coverage["protocol"]
    analysis = protocol["commit_sh_analysis"]
    references = protocol["source_references"]

    def source(reference_key: str) -> str:
        reference = references[reference_key]
        pointer = (
            "evidence/coverage.json#/protocol/source_references/"
            + reference_key
        )
        return _evidence_value(
            _escape(_source_reference_text(reference)),
            pointer,
        )

    def item(
        label: str,
        value: Any,
        value_pointer: str,
        reference_key: str,
    ) -> str:
        return (
            f"<li>{label}: "
            f"{_evidence_value(_escape(str(value)), value_pointer)}"
            f"<br>Source: {source(reference_key)}</li>"
        )

    base = "evidence/coverage.json#/protocol"
    lines = ['<div class="card"><ul>']
    lines.append(item(
        "Default GPUs",
        protocol["num_gpus_default"],
        base + "/num_gpus_default",
        "num_gpus_default",
    ))
    lines.append(item(
        "Device requirement",
        protocol["cuda_device_requirement"],
        base + "/cuda_device_requirement",
        "cuda_device_requirement",
    ))
    lines.append(item(
        "GPU binding",
        protocol["request_gpus_binding"],
        base + "/request_gpus_binding",
        "request_gpus_binding",
    ))
    lines.append(item(
        "Receives NUM_HOURS",
        str(protocol["receives_num_hours"]).lower(),
        base + "/receives_num_hours",
        "receives_num_hours",
    ))
    lines.append(item(
        "Timeout formula",
        protocol["solve_timeout_formula"],
        base + "/solve_timeout_formula",
        "solve_timeout_formula",
    ))
    lines.append(item(
        "Timeout grace minutes",
        protocol["timeout_grace_minutes"],
        base + "/timeout_grace_minutes",
        "timeout_grace_minutes",
    ))
    lines.append(item(
        "Evaluation directories",
        protocol["evaluation_dir_count"],
        base + "/evaluation_dir_count",
        "evaluation_dirs_present",
    ))
    lines.append(item(
        "Active models",
        ", ".join(analysis["current_models_in_arrays"]),
        base + "/commit_sh_analysis/current_models_in_arrays",
        "commit_sh_analysis.current_models_in_arrays",
    ))
    lines.append(item(
        "Active benchmarks",
        ", ".join(analysis["current_benchmarks_in_arrays"]),
        base + "/commit_sh_analysis/current_benchmarks_in_arrays",
        "commit_sh_analysis.current_benchmarks_in_arrays",
    ))
    lines.append(item(
        "MPI hours",
        analysis["htcondor_mpi_is_branch"]["hours"],
        base + "/commit_sh_analysis/htcondor_mpi_is_branch/hours",
        "commit_sh_analysis.htcondor_mpi_is_branch",
    ))
    lines.append(item(
        "MPI GPUs",
        analysis["htcondor_mpi_is_branch"]["gpus"],
        base + "/commit_sh_analysis/htcondor_mpi_is_branch/gpus",
        "commit_sh_analysis.htcondor_mpi_is_branch",
    ))
    lines.append(item(
        "Default-branch ten-hour jobs",
        analysis["htcondor_branch"]["ten_hour_jobs"],
        base + "/commit_sh_analysis/htcondor_branch/ten_hour_jobs",
        "commit_sh_analysis.htcondor_branch",
    ))
    lines.append(item(
        "Default-branch one-hour jobs",
        analysis["htcondor_branch"]["one_hour_jobs"],
        base + "/commit_sh_analysis/htcondor_branch/one_hour_jobs",
        "commit_sh_analysis.htcondor_branch",
    ))
    lines.append("</ul></div>")
    return "\n".join(lines)


def _render_reward_status_summary(reward_hacking: dict[str, Any]) -> str:
    values = [
        (
            "Contamination",
            "training_on_test_sets",
        ),
        (
            "Instruction model",
            "downloading_instruction_tuned_checkpoint",
        ),
        (
            "API misuse",
            "using_discovered_api_key",
        ),
    ]
    return " &middot; ".join(
        label
        + ": "
        + _status_value(
            reward_hacking[key]["status"],
            f"evidence/reward_hacking.json#/{key}/status",
        )
        for label, key in values
    )


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

def render_index_html(
    provenance: dict[str, Any],
    coverage: dict[str, Any],
    reward_hacking: dict[str, Any],
    claims: dict[str, Any],
) -> str:
    """Render the landing page (index.html)."""
    claim1 = claims["claim_1"]
    claim2 = claims["claim_2"]

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="description" content="Reproduction evidence for PostTrainBench (ICML 2026 Agent Repro Challenge)">
<title>PostTrainBench Reproduction</title>
<style>{_CSS}</style>
</head>
<body>
<h1>PostTrainBench Reproduction</h1>
<p>Paper: PostTrainBench: Can LLM Agents Automate LLM Post-Training?</p>
<p>Paper ID: {_evidence_value(
    f'<code>{_escape(provenance["paper_id"])}</code>',
    "evidence/provenance.json#/paper_id",
)} &middot;
Attempt: {_evidence_value(
    f'<code>{_escape(provenance["attempt_id"])}</code>',
    "evidence/provenance.json#/attempt_id",
)}</p>

<nav>
<a href="#claims">Claims</a>
<a href="#coverage">Coverage Matrix</a>
<a href="#protocol">Protocol Audit</a>
<a href="#reward-hacking">Reward Hacking</a>
<a href="#limitations">Limitations</a>
<a href="evidence/manifest.json">Manifest</a>
<a href="report.html">Report</a>
<a href="poster.html">Poster</a>
</nav>

<h2 id="claims">Selected Claims</h2>

{_render_claim(claim1, "claim_1", "Claim 1")}
{_render_claim(claim2, "claim_2", "Claim 2")}

<h2 id="coverage">Coverage Matrix (4 Models &times; 7 Benchmarks)</h2>
{_render_coverage_summary(coverage)}
{_render_matrix(coverage)}

<h2 id="protocol">Protocol Audit</h2>
{_render_protocol(coverage)}

<h2 id="reward-hacking">Reward-Hacking Submodes</h2>
{_render_reward_hacking(reward_hacking)}

<h2 id="limitations">Limitations</h2>
{_render_limitations(claims)}

<footer>
<p>This is not an official challenge verdict. Evidence status is distinct from an official challenge verdict.
{_ptr("evidence/claims.json")}</p>
<p>Paid API cost: USD {provenance["paid_api_cost_usd"]}
{_ptr("evidence/provenance.json#/paid_api_cost_usd")}</p>
</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Report page
# ---------------------------------------------------------------------------

def render_report_html(
    provenance: dict[str, Any],
    coverage: dict[str, Any],
    reward_hacking: dict[str, Any],
    claims: dict[str, Any],
) -> str:
    """Render the detailed report page."""
    claim1 = claims["claim_1"]
    claim2 = claims["claim_2"]

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="description" content="Detailed reproduction report for PostTrainBench">
<title>PostTrainBench Reproduction Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Reproduction Report: PostTrainBench</h1>
<p><a href="index.html">&larr; Back to summary</a></p>

<h2>Provenance</h2>
<div class="card">
<table>
<tr><th>Field</th><th>Value</th><th>Pointer</th></tr>
<tr><td>Paper ID</td><td><code>{provenance["paper_id"]}</code></td><td>{_ptr("evidence/provenance.json#/paper_id")}</td></tr>
<tr><td>Attempt ID</td><td><code>{provenance["attempt_id"]}</code></td><td>{_ptr("evidence/provenance.json#/attempt_id")}</td></tr>
<tr><td>Snapshot</td><td><code>{provenance["assessed_snapshot"][:16]}&#8230;</code></td><td>{_ptr("evidence/provenance.json#/assessed_snapshot")}</td></tr>
<tr><td>Challenge Rev</td><td><code>{provenance["challenge_revision"][:16]}&#8230;</code></td><td>{_ptr("evidence/provenance.json#/challenge_revision")}</td></tr>
<tr><td>Upstream Token</td><td><code>{_escape(provenance["upstream_token"][:40])}&#8230;</code></td><td>{_ptr("evidence/provenance.json#/upstream_token")}</td></tr>
<tr><td>Source Commit</td><td><code>{provenance["source"]["pinned_commit"][:16]}&#8230;</code></td><td>{_ptr("evidence/provenance.json#/source/pinned_commit")}</td></tr>
<tr><td>Dataset Rev</td><td><code>{provenance["dataset"]["pinned_revision"][:16]}&#8230;</code></td><td>{_ptr("evidence/provenance.json#/dataset/pinned_revision")}</td></tr>
<tr><td>Paid API Cost</td><td>USD {provenance["paid_api_cost_usd"]}</td><td>{_ptr("evidence/provenance.json#/paid_api_cost_usd")}</td></tr>
</table>
</div>

<h2>Claim 1: Coverage</h2>
{_render_claim(claim1, "claim_1", "Coverage claim")}

<h3>Coverage Matrix</h3>
{_render_coverage_summary(coverage)}
{_render_matrix(coverage)}

<h3>Protocol Controls</h3>
{_render_protocol(coverage)}

<h2>Claim 2: Reward Hacking</h2>
{_render_claim(claim2, "claim_2", "Reward-hacking claim")}

{_render_reward_hacking(reward_hacking)}

<h2>Limitations</h2>
{_render_limitations(claims)}

<footer>
<p>This evidence is not an official challenge verdict.</p>
<p>Source: {_escape(provenance["source"]["repository"])} at
<code>{provenance["source"]["pinned_commit"][:12]}&#8230;</code> &middot;
License: {provenance["source"]["license"]}</p>
<p>Dataset: {_escape(provenance["dataset"]["repository"])} at
<code>{provenance["dataset"]["pinned_revision"][:12]}&#8230;</code> &middot;
License: {provenance["dataset"]["license"]}</p>
</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Poster page
# ---------------------------------------------------------------------------

def render_poster_html(
    provenance: dict[str, Any],
    coverage: dict[str, Any],
    reward_hacking: dict[str, Any],
    claims: dict[str, Any],
) -> str:
    """Render the poster page."""
    claim1 = claims["claim_1"]
    claim2 = claims["claim_2"]

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="description" content="Reproduction poster for PostTrainBench">
<title>PostTrainBench Reproduction Poster</title>
<style>{_CSS}
body {{ max-width: 900px; }}
.poster-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
@media (max-width: 700px) {{ .poster-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>PostTrainBench: Released-Artifact Audit</h1>
<p>Paper ID: {_evidence_value(
    f'<code>{_escape(provenance["paper_id"])}</code>',
    "evidence/provenance.json#/paper_id",
)} &middot; arXiv: {_evidence_value(
    f'<code>{_escape(provenance["arxiv_id"])}</code>',
    "evidence/provenance.json#/arxiv_id",
)}</p>

<div class="poster-grid">
{_render_claim(claim1, "claim_1", "Claim 1")}
{_render_claim(claim2, "claim_2", "Claim 2")}
</div>

<h2>Coverage Counts</h2>
{_render_coverage_summary(coverage)}

<h2>Coverage Matrix</h2>
{_render_matrix(coverage)}

<h2>Reward-Hacking Statuses</h2>
<div class="card"><p>{_render_reward_status_summary(reward_hacking)}</p></div>

<h2>Limitations</h2>
{_render_limitations(claims)}

<footer>
<p>This evidence is not an official challenge verdict.
{_ptr("evidence/claims.json")}
<a href="index.html">Full evidence summary</a> &middot;
<a href="report.html">Detailed report</a></p>
</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# README with Space frontmatter
# ---------------------------------------------------------------------------

def render_readme(
    provenance: dict[str, Any],
    claims: dict[str, Any],
) -> str:
    """Render README.md with Hugging Face Space frontmatter."""
    claim1 = claims["claim_1"]
    claim2 = claims["claim_2"]

    return f"""\
---
title: PostTrainBench Reproduction
emoji: "\U0001f50d"
colorFrom: blue
colorTo: cyan
sdk: static
app_file: index.html
pinned: false
license: mit
tags:
  - icml2026-repro
  - paper-UnjxMTe57e
---

# PostTrainBench Reproduction

Deterministic CPU-only released-artifact audit for PostTrainBench
(OpenReview: `{provenance["paper_id"]}`, arXiv: `{provenance["arxiv_id"]}`).

## Selected Claims

### Claim 1: {claim1["status"]}

{claim1["text"]}

{claim1["summary"]}

### Claim 2: {claim2["status"]}

{claim2["text"]}

{claim2["summary"]}

## Evidence

- [Evidence summary](index.html)
- [Detailed report](report.html)
- [Poster](poster.html)
- [Provenance](evidence/provenance.json)
- [Coverage](evidence/coverage.json)
- [Reward hacking](evidence/reward_hacking.json)
- [Claims](evidence/claims.json)
- [Manifest](evidence/manifest.json)

## Limitations

This is not an official challenge verdict. See
[the report](report.html) for the full limitation list.

No H100 run is reproduced. A released judge label is not independently
established behavioral truth. The API-key submode remains unavailable.

## Licenses

- Source repository: {provenance["source"]["license"]}
- Dataset: {provenance["dataset"]["license"]}
- Paper: {provenance["paper_license"]}
- This reproduction: MIT

## Cost

Paid API cost: USD {provenance["paid_api_cost_usd"]}
"""
