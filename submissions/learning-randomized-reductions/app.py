import json
from pathlib import Path
import gradio as gr

ROOT = Path(__file__).resolve().parent
EVIDENCE_PATH = ROOT / "evidence/results.json"
PAGES_DIR = ROOT / "pages"

EVIDENCE = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8")) if EVIDENCE_PATH.exists() else {}
PAGE_PATHS = sorted(PAGES_DIR.glob("*.md")) if PAGES_DIR.exists() else []
PAGE_TEXT = [p.read_text(encoding="utf-8") for p in PAGE_PATHS]


def build_claim_rows():
    rows = []
    claims = EVIDENCE.get("claims", [])
    for claim in claims:
        rows.append([
            claim.get("title", ""),
            claim.get("status", ""),
            claim.get("challenge_claim_sha256", "")[:12] + "...",
            claim.get("measured_observation", ""),
        ])
    return rows


with gr.Blocks(title="Learning Randomized Reductions Evidence") as demo:
    gr.Markdown("# Learning Randomized Reductions — Artifact Reproduction Evidence")
    gr.Markdown(
        f"**Paper ID:** `{EVIDENCE.get('paper_id', 'hCAEcqig2C')}` | "
        f"**Attempt ID:** `{EVIDENCE.get('attempt_id', '')}` | "
        f"**Upstream Commit:** `{EVIDENCE.get('upstream_pins', {}).get('git_commit', '')}`"
    )

    if claims := EVIDENCE.get("claims"):
        gr.Dataframe(
            headers=["Claim Title", "Status", "Claim SHA-256", "Measured Observation"],
            value=build_claim_rows(),
            interactive=False,
        )

    for path, text in zip(PAGE_PATHS, PAGE_TEXT, strict=True):
        tab_name = path.stem.split("-", 1)[-1].replace("-", " ").title()
        with gr.Tab(tab_name):
            gr.Markdown(text)

if __name__ == "__main__":
    demo.launch()
