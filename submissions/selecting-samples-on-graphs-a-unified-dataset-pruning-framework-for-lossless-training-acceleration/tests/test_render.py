from __future__ import annotations

import json
from copy import deepcopy
from html import escape
from pathlib import Path

from graph_pruning_repro.render import (
    assert_render_agreement,
    load_accepted_evidence,
    render_distribution_assets,
    render_poster,
    render_report,
    rendered_pointer_values,
    resolve_rfc6901,
)


PROJECT_ROOT = Path(__file__).parents[1]


def test_render_leads_with_two_target_claims_and_boundaries() -> None:
    evidence = json.loads(
        (PROJECT_ROOT / "evidence" / "evidence.json").read_text()
    )
    report = render_report(evidence)
    assert report.index(evidence["target_claims"][0]) < report.index(
        "Unavailable empirical claims"
    )
    assert evidence["target_claims"][1] in report
    assert "appendix_inline_shift_literal" in report
    assert "modular_shift_candidate" in report
    assert "1 then 3" in report
    assert "CIFAR-10/100" in report and "unavailable" in report
    assert "https://arxiv.org/pdf/2606.12913v2" in report


def test_report_numbers_all_come_from_evidence() -> None:
    evidence = json.loads(
        (PROJECT_ROOT / "evidence" / "evidence.json").read_text()
    )
    report = render_report(evidence)
    assert_render_agreement(evidence, report, render_poster(evidence))


def test_all_rendered_json_pointers_are_canonical_and_resolve() -> None:
    evidence = load_accepted_evidence()
    for pointer, displayed_value in rendered_pointer_values(evidence):
        assert "{id}" not in pointer
        assert pointer.startswith("/")
        assert resolve_rfc6901(evidence, pointer) == displayed_value
        for segment in pointer.split("/")[1:]:
            if segment.isdecimal():
                assert segment == "0" or not segment.startswith("0")


def test_notice_and_both_licenses_are_surfaced() -> None:
    assets = render_distribution_assets(load_accepted_evidence())
    assert {
        "NOTICE.md",
        "LICENSE",
        "LICENSES/CC-BY-NC-SA-4.0.txt",
    } <= set(assets)
    notice = assets["NOTICE.md"]
    assert "Dongyue Wu" in notice
    assert "Changxin Gao" in notice


def test_rendering_is_deterministic_and_does_not_mutate_evidence() -> None:
    evidence = load_accepted_evidence()
    original = deepcopy(evidence)
    assert render_report(evidence) == render_report(evidence)
    assert render_poster(evidence) == render_poster(evidence)
    assert evidence == original


def test_every_selected_display_value_has_pointer_metadata() -> None:
    evidence = load_accepted_evidence()
    report = render_report(evidence)
    poster = render_poster(evidence)
    for pointer, value in rendered_pointer_values(evidence):
        assert f"evidence: `{pointer}`" in report
        assert f'data-evidence-path="{pointer}"' in poster
        assert str(value) in report
        assert escape(str(value)) in poster


def test_generated_documents_match_evidence_only_rendering() -> None:
    evidence = load_accepted_evidence()
    report = render_report(evidence)
    poster = render_poster(evidence)
    assert (PROJECT_ROOT / "report.md").read_text() == report
    assert (PROJECT_ROOT / "poster.html").read_text() == poster
    assert (PROJECT_ROOT / "poster_embed.html").read_text() == poster
    readme = (PROJECT_ROOT / "README.md").read_text()
    assert "Unavailable empirical claims" in readme
    assert "NOTICE.md" in readme
    assert "LICENSES/CC-BY-NC-SA-4.0.txt" in readme
    assert "bounded enumeration can refute" in readme
