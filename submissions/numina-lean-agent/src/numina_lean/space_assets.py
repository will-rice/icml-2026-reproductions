"""Render deterministic, source-free static assets from normalized evidence."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any

from numina_lean import UPSTREAM_REVISION


EVIDENCE_FILENAMES = (
    "brascamp_lieb_axioms.json",
    "brascamp_lieb_build.json",
    "claims.json",
    "putnam_axioms.json",
    "putnam_build.json",
)
CLAIM_IDS = ("putnam-12-12", "brascamp-lieb-formalization")
SCOPE = "released-proof verification; not agent re-execution or official verdict"


def _read_normalized_json(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    value = json.loads(raw)
    normalized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if raw != normalized:
        raise ValueError(f"{path.name} is not normalized sorted-key JSON")
    return value


def _load_claims(evidence_dir: Path) -> list[dict[str, Any]]:
    claims = _read_normalized_json(evidence_dir / "claims.json")
    if not isinstance(claims, list) or [
        claim.get("claim_id") for claim in claims if isinstance(claim, dict)
    ] != list(CLAIM_IDS):
        raise ValueError("claims.json must bind exactly the two selected claims")
    if any(
        claim.get("upstream_revision") != UPSTREAM_REVISION for claim in claims
    ):
        raise ValueError("claims.json contains unexpected upstream provenance")
    return claims


def _manifest(evidence_dir: Path) -> dict[str, Any]:
    for filename in EVIDENCE_FILENAMES:
        _read_normalized_json(evidence_dir / filename)
    claims = _load_claims(evidence_dir)
    return {
        "claim_ids": [claim["claim_id"] for claim in claims],
        "evidence_files": {
            filename: {
                "sha256": hashlib.sha256(
                    (evidence_dir / filename).read_bytes()
                ).hexdigest()
            }
            for filename in EVIDENCE_FILENAMES
        },
        "schema_version": 1,
        "scope": SCOPE,
        "upstream_revision": UPSTREAM_REVISION,
    }


def _report(claims: list[dict[str, Any]]) -> str:
    sections = []
    for claim in claims:
        limitations = "\n".join(f"- {item}" for item in claim["limitations"])
        sections.append(
            f"## `{claim['claim_id']}` — partial support\n\n"
            f"**Selected claim:** {claim['claim']}\n\n"
            f"**Computed released-proof observation:** "
            f"{claim['supported_component']}\n\n"
            f"Limitations:\n\n{limitations}"
        )
    return (
        "# Numina-Lean-Agent released-proof verification\n\n"
        "This report provides partial support from released proofs. It is not an "
        "agent rerun and not an official verdict.\n\n"
        + "\n\n".join(sections)
        + "\n\n## Provenance and license boundary\n\n"
        f"`{UPSTREAM_REVISION}`\n\n"
        "The BrascampLieb repository has no LICENSE file. This bundle links to "
        "the pinned repository but does not redistribute its source, caches, "
        "binaries, or raw logs. The agent repository also has no root LICENSE "
        "file, so its source is not redistributed here.\n"
    )


def _html_document(claims: list[dict[str, Any]], *, poster: bool) -> str:
    cards = "\n".join(
        "<article>"
        f"<p class=\"claim-id\">{html.escape(claim['claim_id'])}</p>"
        "<h2>Partial support</h2>"
        f"<p>{html.escape(claim['supported_component'])}</p>"
        "</article>"
        for claim in claims
    )
    title = "Numina-Lean-Agent released-proof verification"
    navigation = (
        ""
        if poster
        else '<nav><a href="report.md">Report</a> · '
        '<a href="manifest.json">Evidence manifest</a></nav>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: #101827; color: #edf4ff; }}
    main {{ max-width: 960px; margin: auto; padding: 3rem 1.5rem; }}
    .scope {{ color: #b8c8df; max-width: 72ch; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(260px,1fr)); gap: 1rem; }}
    article {{ border: 1px solid #385170; border-radius: 14px; padding: 1.25rem; background: #17243a; }}
    h2 {{ color: #ffd166; }}
    .claim-id, code {{ color: #8fd9c7; overflow-wrap: anywhere; }}
    a {{ color: #9fc5ff; }}
  </style>
</head>
<body><main>
  <p class="claim-id">ICML 2026 reproduction evidence</p>
  <h1>{title}</h1>
  <p class="scope">Partial support from deterministic verification of released
  Lean proofs. This is not an agent rerun and not an official verdict.</p>
{navigation}
  <section class="grid">{cards}</section>
  <h2>Limitations and provenance</h2>
  <p>The BrascampLieb and agent repositories have no root LICENSE file. This
  Space does not redistribute their source, caches, binaries, or raw logs.</p>
  <p><code>{html.escape(UPSTREAM_REVISION)}</code></p>
</main></body>
</html>
"""


def render_assets(evidence_dir: Path, output_dir: Path) -> None:
    """Render manifest, report, poster, and Space index from evidence JSON."""
    claims = _load_claims(evidence_dir)
    outputs = {
        "manifest.json": json.dumps(
            _manifest(evidence_dir), indent=2, sort_keys=True
        )
        + "\n",
        "report.md": _report(claims),
        "poster.html": _html_document(claims, poster=True),
        "index.html": _html_document(claims, poster=False),
    }
    for filename, content in outputs.items():
        (output_dir / filename).write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, default=Path("evidence"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    render_assets(args.evidence_dir, args.output_dir)


if __name__ == "__main__":
    main()
