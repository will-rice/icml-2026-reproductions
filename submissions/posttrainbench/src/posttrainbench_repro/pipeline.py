"""Evidence pipeline for PostTrainBench reproduction.

Orchestrates acquisition → audit → serialization → rendering.
Produces deterministic canonical JSON and static HTML.

The ``generate_evidence`` function is the canonical production entry point.
It performs live acquisition, fails closed on any error, and only writes
canonical outputs after all verification passes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from posttrainbench_repro.acquisition import acquire_all
from posttrainbench_repro.audit import (
    audit_protocol,
    audit_reward_hacking,
    compute_coverage,
    evaluate_claims,
    get_provenance,
)
from posttrainbench_repro.constants import CANONICAL_OUTPUTS
from posttrainbench_repro.render import (
    render_index_html,
    render_poster_html,
    render_readme,
    render_report_html,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _canonical_json(obj: Any) -> str:
    """Serialize to canonical JSON: sorted keys, 2-space indent, trailing newline."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def run_pipeline(
    acquired: dict[str, Any],
    *,
    output_root: Path | None = None,
) -> dict[str, Path]:
    """Run the complete evidence pipeline from pre-acquired data.

    Requires a verified ``acquired`` dict from :func:`acquisition.acquire_all`.
    Returns a mapping of relative path → absolute path for all outputs.

    This function writes canonical outputs only after all computation
    succeeds — fail-closed behavior.
    """
    github = acquired["github"]
    destination = PROJECT_ROOT if output_root is None else Path(output_root)

    # 1. Provenance
    provenance = get_provenance(acquired)

    # 2. Coverage
    coverage = compute_coverage(acquired["hf_inventory"])

    # 3. Protocol audit (reuses once-fetched blob contents)
    protocol = audit_protocol(
        github["blob_contents"],
        github["entries"],
    )

    # Merge protocol into coverage for the canonical output
    coverage_output = {**coverage, "protocol": protocol}

    # 4. Reward-hacking audit
    reward_hacking = audit_reward_hacking(acquired)

    # 5. Claims (requires verified audit data)
    claims = evaluate_claims(coverage, protocol, reward_hacking)

    # 6. Compute all outputs in memory before writing anything (fail-closed)
    outputs: dict[str, str] = {}

    outputs["evidence/provenance.json"] = _canonical_json(provenance)
    outputs["evidence/coverage.json"] = _canonical_json(coverage_output)
    outputs["evidence/reward_hacking.json"] = _canonical_json(reward_hacking)
    outputs["evidence/claims.json"] = _canonical_json(claims)
    outputs["index.html"] = render_index_html(
        provenance, coverage_output, reward_hacking, claims
    )
    outputs["report.html"] = render_report_html(
        provenance, coverage_output, reward_hacking, claims
    )
    outputs["poster.html"] = render_poster_html(
        provenance, coverage_output, reward_hacking, claims
    )
    outputs["README.md"] = render_readme(provenance, claims)

    # 7. Manifest (hashes every other output)
    manifest: dict[str, dict[str, Any]] = {}
    for rel_path in sorted(outputs):
        content_bytes = outputs[rel_path].encode("utf-8")
        manifest[rel_path] = {
            "sha256": hashlib.sha256(content_bytes).hexdigest(),
            "size": len(content_bytes),
        }
    outputs["evidence/manifest.json"] = _canonical_json(manifest)

    # 8. Transactional output publication: stage, then replace.
    #    If any write/replace fails, restore all original bytes and remove
    #    newly created outputs.
    evidence_dir = destination / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Save originals for rollback
    originals: dict[str, bytes | None] = {}
    for rel_path in outputs:
        out_path = destination / rel_path
        if out_path.exists():
            originals[rel_path] = out_path.read_bytes()
        else:
            originals[rel_path] = None  # marks as newly created

    written: list[str] = []
    try:
        for rel_path, content in outputs.items():
            out_path = destination / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            written.append(rel_path)
    except Exception:
        # Rollback: restore originals, remove newly created files
        for rel_path in written:
            out_path = destination / rel_path
            if originals[rel_path] is not None:
                out_path.write_bytes(originals[rel_path])
            elif out_path.exists():
                out_path.unlink()
        raise

    # Return output paths
    result: dict[str, Path] = {}
    for rel_path in CANONICAL_OUTPUTS:
        result[rel_path] = destination / rel_path
    return result


def generate_evidence(
    *,
    output_root: Path | None = None,
) -> dict[str, Path]:
    """Canonical production entry point: acquire → verify → emit.

    Performs live acquisition from pinned GitHub commit and HF dataset
    revision, then runs the full pipeline.  Fails closed — if any step
    fails, no canonical outputs are produced or replaced.
    """
    acquired = acquire_all()
    return run_pipeline(acquired, output_root=output_root)


def main() -> None:
    """CLI entry point for ``posttrainbench-evidence`` script."""
    print("PostTrainBench evidence pipeline")
    print("================================")
    print()
    print("Acquiring pinned artifacts...")
    outputs = generate_evidence()
    print()
    print("Evidence generated successfully:")
    for rel, path in sorted(outputs.items()):
        print(f"  {rel}: {path}")
    print()
    print("All canonical outputs verified and written.")


if __name__ == "__main__":
    main()
