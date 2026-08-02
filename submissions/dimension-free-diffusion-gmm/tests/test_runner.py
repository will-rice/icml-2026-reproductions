from __future__ import annotations

import time
from pathlib import Path

import pytest

from diffusion_gmm_repro.claims import LIVE_CLAIMS, validate_claim_text
from diffusion_gmm_repro.runner import (
    ExperimentConfig,
    assemble_bundle,
    cell_id,
    run_cells,
)


def test_all_live_claim_texts_match_pinned_digests() -> None:
    assert [claim.digest for claim in LIVE_CLAIMS] == [
        "eecfa1e84d2a7bcf89795cb1827c7296f91190acf22b9be19386119a5c4b9d4d",
        "cd8012bbf8bded49e76c53bb4c456b4c8fdda05fbead3ea99760e6c4406b53b0",
        "3791bd9209d94bce1dd50fadd9d43244340dc75b7593c7e5adf4828b0e3ddf35",
        "43fed73be7d63a64905f82e9e6cab5859723fe96bb537c7100c9be385c0d2623",
        "2ee4da9763c5432ca007f024fa10acc8101e81adc817baa0d6a88012dec19bb8",
    ]


def test_validate_claim_text_valid_and_invalid() -> None:
    claim = validate_claim_text(LIVE_CLAIMS[0].digest)
    assert claim.digest == LIVE_CLAIMS[0].digest
    assert claim.text == LIVE_CLAIMS[0].text
    with pytest.raises(ValueError, match="invalid claim text or digest"):
        validate_claim_text("unknown claim text")


def test_cell_id_is_canonical_and_order_independent() -> None:
    assert cell_id({"seed": 2, "steps": 128}) == cell_id({"steps": 128, "seed": 2})


def test_assembly_rejects_incomplete_required_cells(tmp_path: Path) -> None:
    config = ExperimentConfig.pilot()
    with pytest.raises(ValueError, match="missing required cells"):
        assemble_bundle(config, cells_dir=tmp_path)


def test_runner_stops_before_resource_deadline(tmp_path: Path) -> None:
    result = run_cells(
        ExperimentConfig.pilot(),
        output_dir=tmp_path,
        deadline_monotonic=time.monotonic() - 1.0,
        max_rss_bytes=16 * 1024**3,
    )
    assert result["status"] == "resource-cap"
    assert result["unrun_cells"]
