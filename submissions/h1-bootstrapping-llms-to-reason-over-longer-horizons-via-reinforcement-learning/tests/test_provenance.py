from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_evidence import generate_evidence


def test_provenance():
    bundle = generate_evidence()
    prov = bundle["provenance"]
    assert "source_urls" in prov
    assert len(prov["source_urls"]) >= 2
    assert "sha256_digests" in prov
