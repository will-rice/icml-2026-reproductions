from fac_evidence.bundle import build_evidence_bundle


def render_report() -> str:
    bundle = build_evidence_bundle()
    manifest = bundle["manifest"]
    lines = [
        "# FAC Synthesis Reproduction Report",
        "",
        f"Paper: {manifest['paper_id']} - {manifest['paper_title']}",
        f"Attempt: {manifest['attempt_id']}",
        f"Snapshot: {manifest['snapshot_id']}",
        "",
        "## Upstream Pins",
        "",
        f"- arXiv source SHA-256: `{manifest['artifacts']['arxiv_source']['sha256']}`",
        f"- GitHub revision: `{manifest['artifacts']['github']['revision']}`",
        f"- HF demo revision: `{manifest['artifacts']['hf_demo']['revision']}`",
        f"- HF dataset API access: `{manifest['artifacts']['hf_dataset_api']['access']}` (HTTP {manifest['artifacts']['hf_dataset_api']['status_code']})",
        "",
        "## Claim Results",
        "",
    ]
    for claim in bundle["claims"]:
        lines.extend(
            [
                f"### {claim['claim_sha256']}",
                "",
                f"Status: `{claim['status']}`",
                "",
                claim["claim"],
                "",
                claim["evidence"],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
