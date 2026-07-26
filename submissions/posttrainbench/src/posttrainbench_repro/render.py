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
        for c in counts:
            rows.append(f"<td>{c}</td>")
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
    parts.append('<div class="card">')
    parts.append(f'<h3>Training on Test Sets {_status_span(contam["status"])}</h3>')
    parts.append(f'<p>{_ptr("evidence/reward_hacking.json#/training_on_test_sets")}</p>')
    parts.append(f'<p>Witness: <code>{_escape(contam["witness_path"])}</code></p>')
    parts.append(f'<p>SHA-256: <code>{contam["witness_sha256"]}</code></p>')
    parts.append(f'<p>Content: <code>{_escape(contam["witness_bytes"])}</code></p>')
    parts.append(f'<p>Type: {contam["observation_type"]}</p>')
    parts.append("</div>")

    # Mode 2: Instruction-model
    instr = rh["downloading_instruction_tuned_checkpoint"]
    parts.append('<div class="card">')
    parts.append(f'<h3>Instruction-Tuned Checkpoint {_status_span(instr["status"])}</h3>')
    parts.append(f'<p>{_ptr("evidence/reward_hacking.json#/downloading_instruction_tuned_checkpoint")}</p>')
    parts.append(f'<p>Judgment: <code>{_escape(instr["judgment_path"])}</code></p>')
    parts.append(f'<p>Judgment SHA-256: <code>{instr["judgment_sha256"]}</code></p>')
    parts.append(f'<p>Content: <code>{_escape(instr["judgment_bytes"])}</code> (no trailing newline)</p>')
    parts.append("<h4>Safe Trace Excerpts</h4>")
    for exc in instr.get("safe_excerpts", []):
        parts.append(f'<div class="excerpt">Record {exc["record"]} {exc["json_pointer"]}: {_escape(str(exc["text"]))}\nSHA-256: {exc["sha256"]}</div>')
    parts.append(f'<p>Complete trace ({instr["trace"]["size"]:,} bytes) is NOT redistributed.</p>')
    parts.append("</div>")

    # Mode 3: API misuse
    api = rh["using_discovered_api_key"]
    parts.append('<div class="card">')
    parts.append(f'<h3>API Key Discovery {_status_span(api["status"])}</h3>')
    parts.append(f'<p>{_ptr("evidence/reward_hacking.json#/using_discovered_api_key")}</p>')
    parts.append(f'<p>{_escape(api["unavailability_reason"])}</p>')
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
        for lim in claim.get("limitations", []):
            if lim not in seen:
                seen.add(lim)
                parts.append(f'<div class="limitation">{_escape(lim)}</div>')
    return "\n".join(parts)


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
<p>Paper: PostTrainBench: Can LLM Agents Automate LLM Post-Training?
{_ptr("evidence/provenance.json#/paper_id")}</p>
<p>Paper ID: <code>{provenance["paper_id"]}</code> &middot;
Attempt: <code>{provenance["attempt_id"]}</code></p>

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

<div class="card">
<h3>Claim 1 {_status_span(claim1["status"])}</h3>
<p>{_escape(claim1["text"])}</p>
<p>{_ptr("evidence/claims.json#/claim_1")}</p>
<p>SHA-256: <code>{claim1["sha256"]}</code></p>
<p>{_escape(claim1["summary"])}</p>
</div>

<div class="card">
<h3>Claim 2 {_status_span(claim2["status"])}</h3>
<p>{_escape(claim2["text"])}</p>
<p>{_ptr("evidence/claims.json#/claim_2")}</p>
<p>SHA-256: <code>{claim2["sha256"]}</code></p>
<p>{_escape(claim2["summary"])}</p>
</div>

<h2 id="coverage">Coverage Matrix (4 Models &times; 7 Benchmarks)</h2>
<p>{_ptr("evidence/coverage.json#/cell_counts")}</p>
<p>Recognized tasks: {coverage["recognized_task_count"]}
{_ptr("evidence/coverage.json#/recognized_task_count")}</p>
<p>Recognized roots: {coverage.get("recognized_root_count", "N/A")}
&middot; Root/cell pairs: {coverage.get("recognized_root_cell_pairs", "N/A")}
&middot; Duplicate pairs: {coverage.get("duplicate_job_pairs", "N/A")}
&middot; Missing pairs: {coverage.get("missing_root_cell_pairs", "N/A")}</p>
{_render_matrix(coverage)}

<h2 id="protocol">Protocol Audit</h2>
<p>{_ptr("evidence/coverage.json#/protocol")}</p>
<div class="card">
<ul>
<li>GPU default: {coverage.get("protocol", {}).get("num_gpus_default", 1)} {_ptr("evidence/coverage.json#/protocol/num_gpus_default")}</li>
<li>Device: <code>{_escape(str(coverage.get("protocol", {}).get("cuda_device_requirement", "")))}</code></li>
<li>Binding: <code>{_escape(str(coverage.get("protocol", {}).get("request_gpus_binding", "")))}</code></li>
<li>Timeout: {coverage.get("protocol", {}).get("solve_timeout_formula", "")} (5-min grace)</li>
<li>Eval dirs: {len(coverage.get("protocol", {}).get("evaluation_dirs_present", []))} of 7 found</li>
</ul>
</div>

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

<h2>Claim 1: Coverage {_status_span(claim1["status"])}</h2>
<div class="card">
<p>{_escape(claim1["text"])}</p>
<p>SHA-256: <code>{claim1["sha256"]}</code></p>
<p>{_escape(claim1["summary"])}</p>
</div>

<h3>Coverage Matrix</h3>
{_render_matrix(coverage)}
<p>Tasks: {coverage["recognized_task_count"]} &middot;
Roots: {coverage.get("recognized_root_count", "N/A")} &middot;
Pairs: {coverage.get("recognized_root_cell_pairs", "N/A")} &middot;
Duplicates: {coverage.get("duplicate_job_pairs", "N/A")} &middot;
Missing: {coverage.get("missing_root_cell_pairs", "N/A")}</p>

<h3>Protocol Controls</h3>
<div class="card">
<ul>
<li>Default GPUs: {coverage.get("protocol", {}).get("num_gpus_default", 1)}</li>
<li>Device requirement: <code>{_escape(str(coverage.get("protocol", {}).get("cuda_device_requirement", "")))}</code></li>
<li>GPU binding: <code>{_escape(str(coverage.get("protocol", {}).get("request_gpus_binding", "")))}</code></li>
<li>Timeout formula: {coverage.get("protocol", {}).get("solve_timeout_formula", "")} minutes (5-minute grace)</li>
<li>Evaluation directories: {len(coverage.get("protocol", {}).get("evaluation_dirs_present", []))} of 7</li>
</ul>
</div>

<h2>Claim 2: Reward Hacking {_status_span(claim2["status"])}</h2>
<div class="card">
<p>{_escape(claim2["text"])}</p>
<p>SHA-256: <code>{claim2["sha256"]}</code></p>
<p>{_escape(claim2["summary"])}</p>
</div>

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
    contam = reward_hacking["training_on_test_sets"]
    instr = reward_hacking["downloading_instruction_tuned_checkpoint"]
    api = reward_hacking["using_discovered_api_key"]

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
<p>Paper ID: <code>{provenance["paper_id"]}</code> &middot; arXiv: {provenance["arxiv_id"]}
{_ptr("evidence/provenance.json#/paper_id")}</p>

<div class="poster-grid">
<div class="card">
<h3>Claim 1 {_status_span(claim1["status"])}</h3>
<p>{_escape(claim1["text"])}</p>
<p>{_ptr("evidence/claims.json#/claim_1/status")}</p>
<p><strong>{coverage["recognized_task_count"]}</strong> tasks across
<strong>{len(coverage["accepted_benchmarks"])}</strong> benchmarks &times;
<strong>{len(coverage["accepted_models"])}</strong> models
{_ptr("evidence/coverage.json#/recognized_task_count")}</p>
<p>Roots: {coverage.get("recognized_root_count", "N/A")} &middot;
Pairs: {coverage.get("recognized_root_cell_pairs", "N/A")} &middot;
Duplicates: {coverage.get("duplicate_job_pairs", "N/A")}
{_ptr("evidence/coverage.json#/duplicate_job_pairs")}</p>
</div>

<div class="card">
<h3>Claim 2 {_status_span(claim2["status"])}</h3>
<p>{_escape(claim2["text"])}</p>
<p>{_ptr("evidence/claims.json#/claim_2/status")}</p>
<p>Contamination: {_status_span(contam["status"])}
{_ptr("evidence/reward_hacking.json#/training_on_test_sets/status")} &middot;
Instruction model: {_status_span(instr["status"])}
{_ptr("evidence/reward_hacking.json#/downloading_instruction_tuned_checkpoint/status")} &middot;
API misuse: {_status_span(api["status"])}
{_ptr("evidence/reward_hacking.json#/using_discovered_api_key/status")}</p>
</div>
</div>

<h2>Coverage Matrix</h2>
<p>{_ptr("evidence/coverage.json#/cell_counts")}</p>
{_render_matrix(coverage)}

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
