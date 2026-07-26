import hashlib
import pytest
from recurrent_sampler_repro.evidence import (
    CLAIM_1_TEXT,
    CLAIM_1_SHA256,
    CLAIM_2_TEXT,
    CLAIM_2_SHA256,
)


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_claim_1_binding():
    assert compute_sha256(CLAIM_1_TEXT) == CLAIM_1_SHA256
    assert CLAIM_1_SHA256 == "d0da87ee16f7485d3dff369e7465f66299c55ac003a54e1cf8c00b3a0ad8b265"


def test_claim_2_binding():
    assert compute_sha256(CLAIM_2_TEXT) == CLAIM_2_SHA256
    assert CLAIM_2_SHA256 == "2e15221c8b5516b0ab705e29a3d7c5d924ed5f0187c970a0caf60a1402757804"


def test_claim_mutation_rejection():
    # Test whitespace change
    mutated_c1 = CLAIM_1_TEXT + " "
    assert compute_sha256(mutated_c1) != CLAIM_1_SHA256

    # Test punctuation change
    mutated_c2 = CLAIM_2_TEXT.replace("Theorem 4.2", "Theorem 4.2.")
    assert compute_sha256(mutated_c2) != CLAIM_2_SHA256
