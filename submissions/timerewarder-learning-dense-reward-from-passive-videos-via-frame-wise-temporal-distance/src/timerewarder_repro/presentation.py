"""Verified, single-source presentation of the canonical evidence bundle."""

from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import cast

from timerewarder_repro.evidence import measurement_sha256


def load_verified_evidence(path: Path) -> dict[str, object]:
    """Load evidence only when its stable content matches its recorded hash."""
    import json

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("evidence is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError("evidence must be a mapping")
    recorded = value.get("measurement_sha256")
    if not isinstance(recorded, str) or measurement_sha256(value) != recorded:
        raise ValueError("measurement hash mismatch")
    claims = value.get("claims")
    if not isinstance(claims, list) or len(claims) != 6:
        raise ValueError("evidence must contain six claims")
    return cast(dict[str, object], value)


def claim_rows(bundle: Mapping[str, object]) -> list[list[str]]:
    """Return stable display rows: label, status, evidence, limitation."""
    claims = bundle.get("claims")
    if not isinstance(claims, list) or len(claims) != 6:
        raise ValueError("evidence must contain six claims")
    rows = []
    for index, record in enumerate(claims, start=1):
        if not isinstance(record, Mapping):
            raise ValueError("claim must be a mapping")
        rows.append(
            [
                f"Claim {index}",
                str(record.get("status", "")),
                str(record.get("evidence", "")),
                str(record.get("limitations", "")),
            ]
        )
    return rows


def render_poster(bundle: Mapping[str, object]) -> str:
    """Render a self-contained poster from canonical measurements."""
    revisions = _mapping(_mapping(bundle["inputs"])["revisions"])
    protocol = _mapping(bundle["protocol"])
    measurements = _mapping(bundle["measurements"])
    representative = _mapping(measurements["representative"])
    pooled = _mapping(representative["pooled_metrics"])
    theory = _mapping(measurements["theory"])
    measurement_hash = escape(str(bundle["measurement_sha256"]))
    rows = claim_rows(bundle)
    claim_cards = "\n".join(
        (
            f'<article class="claim {escape(status)}">'
            f"<h3>{escape(label)} · {escape(status)}</h3>"
            f"<p>{escape(evidence)}</p>"
            f"<p><strong>Limitation:</strong> {escape(limitation)}</p>"
            "</article>"
        )
        for label, status, evidence, limitation in rows
    )
    pins = "".join(
        f"<li><strong>{escape(str(name))}:</strong> {escape(str(revision))}</li>"
        for name, revision in revisions.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TimeRewarder reproduction evidence</title>
<style>
body {{ margin: 0; font: 16px/1.45 system-ui, sans-serif; color: #10233a;
  background: #eef4fb; }}
main {{ max-width: 1120px; margin: auto; padding: 2rem; }}
h1 {{ font-size: 2.3rem; margin-bottom: .2rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(280px,1fr));
  gap: 1rem; }}
section, .claim {{ background: white; border-radius: 12px; padding: 1rem;
  border: 1px solid #bed1e5; }}
.claim.partial {{ border-left: 8px solid #e9a600; }}
.claim.unavailable {{ border-left: 8px solid #b33a3a; background: #fff7f7; }}
code {{ overflow-wrap: anywhere; }}
</style>
</head>
<body><main>
<h1>TimeRewarder: independently executable evidence</h1>
<p>Released artifacts were recomputed; paper-reported values are not presented
as reproduced measurements.</p>
<div class="grid">
<section><h2>Scope and immutable inputs</h2><ul>{pins}</ul></section>
<section><h2>Safety boundary</h2><p>Legacy weights were converted in an isolated,
bounded process and independently approved. Runtime inference accepts only
approved safetensors artifacts.</p></section>
<section><h2>Representative protocol</h2>
<p><strong>Figure 3 diagnostic:</strong> five-video-per-task released-model
protocol, not the paper's full comparative Figure 3 protocol.</p>
<p>{escape(str(protocol["task_count"]))} tasks ×
{escape(str(protocol["videos_per_task"]))} videos ×
{escape(str(protocol["ordered_pairs_per_video"]))} ordered pairs.</p></section>
<section><h2>Temporal-distance measurements</h2>
<ul><li>Prediction MAE: {escape(str(pooled["prediction_mae"]))}</li>
<li>Zero-baseline MAE: {escape(str(pooled["zero_baseline_mae"]))}</li>
<li>Relative improvement: {escape(str(pooled["relative_improvement"]))}</li>
<li>Sign accuracy: {escape(str(pooled["sign_accuracy"]))}</li>
<li>Mean antisymmetry error: {escape(str(pooled["mean_antisymmetry_error"]))}</li>
<li>Released-model mean VOC: {escape(str(representative["mean_voc"]))}</li></ul>
</section>
<section><h2>Theory audit</h2>
<p>Finite Bellman recurrences and the gamma-one temporal-distance identity were
checked (maximum absolute residual:
{escape(str(theory["max_absolute_bellman_residual"]))}). Assumptions include full
observability, deterministic transitions, an optimal trajectory, a terminal
goal, and unaliased observations. An aliasing counterexample is retained.</p>
</section>
<section><h2>Deterministic fixture</h2><p>The passive-pair fixture is
<strong>diagnostic-only</strong>; it is not a paper-scale acceptance result.</p>
</section></div>
<h2>Six challenge claims</h2><div class="grid">{claim_cards}</div>
<section><h2>Reproduce</h2>
<pre><code>uv run pytest -q
uv run timerewarder-repro fixture
uv run timerewarder-repro build-evidence --manifest artifacts/manifest.json \
--acquisition artifacts/acquisition.json --registry artifacts/checkpoints.json \
--source-root artifacts/source --representative artifacts/representative.json \
--output artifacts/evidence.json</code></pre>
<p><strong>Measurement SHA-256:</strong> <code>{measurement_hash}</code></p>
</section>
</main></body></html>
"""


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("expected mapping")
    return value
