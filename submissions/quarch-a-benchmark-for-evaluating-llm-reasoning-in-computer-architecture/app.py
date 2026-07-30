import json
from pathlib import Path
import gradio as gr

def load_bundle():
    bundle_path = Path(__file__).parent / "evidence" / "bundle.json"
    if bundle_path.exists():
        with open(bundle_path, encoding="utf-8") as f:
            return json.load(f)
    return {}

bundle = load_bundle()

def get_summary():
    if not bundle:
        return "Bundle not generated yet."
    output = f"# Reproduction Evidence: {bundle.get('title', 'QuArch')}\n\n"
    output += f"**Paper ID:** `{bundle.get('paper_id')}`  \n"
    output += f"**Upstream Pin:** `{bundle.get('upstream_pin')}`  \n"
    output += f"**Total QA Pairs:** `{bundle.get('total_qa_pairs')}`  \n\n"
    output += "## Claim Verification Results\n\n"
    for claim in bundle.get("claims", []):
        output += f"### Claim {claim['claim_id']} ({claim['status'].upper()})\n"
        output += f"- **Text:** {claim['claim_text']}\n"
        output += f"- **Observation:** {claim['observation']}\n"
        output += f"- **SHA-256:** `{claim['claim_sha256']}`\n\n"
    return output

with gr.Blocks(title="QuArch Reproduction") as demo:
    gr.Markdown(get_summary())

if __name__ == "__main__":
    demo.launch()
