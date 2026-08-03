from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_evidence import build_evidence


def test_mxgraph_sample_parses():
    bundle = build_evidence()
    sample_xml = bundle["observations"]["mxgraph_xml"]["sample_xml"]
    root = ET.fromstring(sample_xml)
    assert root.tag == "mxGraphModel"
    assert len(root.findall(".//mxCell")) >= 3


def test_dataset_counts_match_claim():
    bundle = build_evidence()
    counts = bundle["observations"]["dataset_counts"]
    assert counts["rows"] == 1449
    assert counts["coarse_domains"] == 6
    assert counts["subdomains"] == 15
    assert counts["non_empty_xml"] == 1444


def test_task_protocols_are_distinct():
    bundle = build_evidence()
    protocols = bundle["observations"]["task_protocols"]
    assert protocols["vision_to_code"]["input"] == ["image", "structured_caption"]
    assert protocols["vision_to_code"]["output"] == "complete_mxgraph_xml"
    assert protocols["code_to_code_editing"]["input"] == [
        "source_xml",
        "rendered_image",
        "instruction",
    ]
    assert protocols["code_to_code_editing"]["output"] == "json_xml_patch"


def test_bundle_file_round_trips(tmp_path):
    from generate_evidence import main

    output = tmp_path / "bundle.json"
    main(["--output", str(output)])
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["paper_id"] == "XjSd2CtV20"
    assert set(data["claim_results"]) == {"claim-1", "claim-2", "claim-3"}
