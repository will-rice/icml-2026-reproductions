from pathlib import Path
import pytest

EXPECTED_PAGE_NAMES = [
    "00-summary.md",
    "01-correlated-sampling-theory.md",
    "02-rsr-bench-census.md",
    "03-vanilla-and-sigmoid.md",
    "04-agentic-bitween.md",
    "05-nonlinear-invariant-falsification.md",
    "06-methods-and-provenance.md",
]

EXPECTED_HASHES = [
    "5f0d21d91c0ae1d2877563e7115e804db60361304db4aea72b97596300e60f57",
    "79d94d106cfded95104c54624068a07dc9ae16dca681a6ad5370bbb648e8c7de",
    "4b8bfdf084cb0038acc0a589837dc4379ba1fb079f30f4be8edf839a21d23a51",
    "9b35061b3b4e2873f1b7a4fffc6fa22d659f281c096d990706ebd805303c4c00",
    "13999601811ffe2bb8e9526ed601e9d59480b217d6d1917787db2a9c7dbc8372",
]


def test_root_pages_cover_every_claim(project_root):
    pages = sorted((project_root / "pages").glob("*.md"))
    assert [page.name for page in pages] == EXPECTED_PAGE_NAMES
    text = "\n".join(page.read_text(encoding="utf-8") for page in pages)
    for digest in EXPECTED_HASHES:
        assert digest in text
    assert (
        "falsified"
        in (project_root / "pages/05-nonlinear-invariant-falsification.md")
        .read_text(encoding="utf-8")
        .lower()
    )
