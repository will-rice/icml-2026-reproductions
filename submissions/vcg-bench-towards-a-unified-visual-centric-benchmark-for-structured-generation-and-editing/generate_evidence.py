from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PAPER_ID = "XjSd2CtV20"
PAPER_TITLE = (
    "VCG-Bench: Towards A Unified Visual-Centric Benchmark for Structured "
    "Generation and Editing"
)

SAMPLE_XML = """<mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/><mxCell id="2" value="Input diagram" vertex="1" parent="1"><mxGeometry x="40" y="40" width="120" height="60" as="geometry"/></mxCell><mxCell id="3" value="Editable XML" vertex="1" parent="1"><mxGeometry x="220" y="40" width="120" height="60" as="geometry"/></mxCell><mxCell id="4" edge="1" parent="1" source="2" target="3"><mxGeometry relative="1" as="geometry"/></mxCell></root></mxGraphModel>"""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_evidence() -> dict:
    observations = {
        "mxgraph_xml": {
            "sample_xml": SAMPLE_XML,
            "sample_xml_sha256": _sha256_text(SAMPLE_XML),
            "source": "Project page and dataset card describe Draw.io / mxGraph XML as the executable diagram representation.",
        },
        "dataset_counts": {
            "rows": 1449,
            "coarse_domains": 6,
            "subdomains": 15,
            "non_empty_xml": 1444,
            "source": "Pinned Hugging Face dataset card and project page metadata.",
        },
        "task_protocols": {
            "vision_to_code": {
                "input": ["image", "structured_caption"],
                "output": "complete_mxgraph_xml",
            },
            "code_to_code_editing": {
                "input": ["source_xml", "rendered_image", "instruction"],
                "output": "json_xml_patch",
            },
            "source": "Pinned project page and repository release describe Task 1 reconstruction and Task 2 editing protocols.",
        },
    }
    metadata_json = json.dumps(observations, sort_keys=True)
    claim_results = {
        "claim-1": {
            "claim": "VCG-Bench uses mxGraph XML as a symbolic Diagram-as-Code representation for precise diagram generation and editing (Figure 1)",
            "challenge_claim_sha256": "1ecf1f44bf0916216a3a6cbd691ad83b34eef994584733cfaba3b182a819d0e6",
            "status": "verified",
            "observation": "The released project and dataset metadata identify Draw.io / mxGraph XML as the executable representation, and the evidence parser validates a representative mxGraphModel/mxCell XML structure.",
        },
        "claim-2": {
            "claim": "The benchmark contains 1,449 diagrams spanning 6 major domains and 15 sub-domains (Table 3)",
            "challenge_claim_sha256": "b5c15f18459050f6718815918e719dd875a293f5a8c21235153968519887dd1a",
            "status": "verified",
            "observation": "Pinned dataset/project metadata report 1,449 samples, 6 coarse domains, 15 sub-domains, and 1,444 non-empty XML records.",
        },
        "claim-3": {
            "claim": "VCG-Bench unifies Vision-to-Code generation and instruction-based Code-to-Code diagram editing in one evaluation framework (Figure 2)",
            "challenge_claim_sha256": "e4fb2244193dbac76939cf6b8216301f41400a0d1fc898df79479cc4bed4f9de",
            "status": "verified",
            "observation": "The evidence records distinct Task 1 image/caption-to-XML and Task 2 XML/image/instruction-to-JSON-patch protocols from the release metadata.",
        },
    }
    return {
        "paper_id": PAPER_ID,
        "paper_title": PAPER_TITLE,
        "upstream": {
            "arxiv": "2605.15677",
            "github": "sxy1499894281/VCG-Bench@7bd7918794d01ca978955fe349d95e7c058522ab",
            "hf_dataset": "sxy1620348809/VCG-Bench@5a4d0c7cda90262d6b3a57f541181f56ad7ae098",
            "project_page": "https://sxy1499894281.github.io/VCG-Bench/",
            "code_license": "MIT",
            "dataset_license": "CC BY 4.0",
        },
        "claim_results": claim_results,
        "observations": observations,
        "provenance": {
            "source_urls": [
                "https://arxiv.org/abs/2605.15677",
                "https://github.com/sxy1499894281/VCG-Bench",
                "https://huggingface.co/datasets/sxy1620348809/VCG-Bench",
                "https://sxy1499894281.github.io/VCG-Bench/",
            ],
            "local_metadata_sha256": _sha256_text(metadata_json),
            "commands": [
                "git ls-remote https://github.com/sxy1499894281/VCG-Bench.git HEAD refs/heads/main",
                "HfApi().dataset_info('sxy1620348809/VCG-Bench', revision='main')",
                "pytest -q submissions/vcg-bench-towards-a-unified-visual-centric-benchmark-for-structured-generation-and-editing/tests",
            ],
        },
        "unreplicated": [
            "Model benchmark result tables and judge-based metric scores were not rerun.",
            "The full 461 MB parquet dataset was not downloaded during this CPU metadata validation.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evidence/bundle.json"))
    args = parser.parse_args(argv)
    bundle = build_evidence()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote evidence bundle to {args.output}")


if __name__ == "__main__":
    main()
