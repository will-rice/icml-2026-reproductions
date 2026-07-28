from dataclasses import dataclass
import json
from pathlib import Path

PAPER_ID = "7UEBX1KU1y"
ATTEMPT_ID = "933665ed-b7ed-4d73-9b07-35704660a184"
SNAPSHOT_ID = (
    "09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199"
)
UPSTREAM_REVISION = "arxiv:2605.20834v1"
LIVE_CLAIMS = (
    "The paper proves DPO-RLHF equivalence is conditional on the RLHF-optimal "
    "policy preferring human-preferred responses (Section 3).",
    "When the equivalence assumption fails, DPO optimizes relative advantage "
    "over the reference policy rather than absolute human-preference alignment "
    "(Section 3).",
    "The paper characterizes undesirable solution spaces in which policies "
    "reduce DPO loss while preferring dispreferred responses (Section 3).",
    "Constrained Preference Optimization augments RLHF with constraints and "
    "derives a stationary DPO-like loss with an adaptive reference-based margin "
    "(Section 4.3).",
    "The paper gives a soft-margin ranking interpretation showing DPO can "
    "implement margin ranking with potentially negative targets (Section 5).",
    "Experiments on standard benchmarks report state-of-the-art performance "
    "for CPO (Section 6).",
)
LIVE_CLAIM_HASHES = (
    "588c9334124771dc2ff7fc51494f4328329ab13dc21d4522a0e91b6f6417240a",
    "4820743d0eac6cc30b4a75d2be41f49193b0ea4ad4168bea2200a9f16cc77a86",
    "6c26fe711e2f10b44cb933b89b12982fef3cf3bcc760668a0b0fa9d15e1965dc",
    "a80267886061211c131041549df22264e0c713a9759a76f0ab37bac69a436af1",
    "7d797875f18478f305a8dc08d860a29ba4f15c3b97fb4c9d41e55363975553be",
    "8df1fece656f02adbdf85fb78bc8993591f1abc9ee78c957388ab4b4eac37dcd",
)


@dataclass(frozen=True)
class ClaimBinding:
    challenge_claim: str
    challenge_claim_sha256: str
    target_claim: str | None
    targeted: bool
    equations: tuple[str, ...]


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_source_record(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if value["paper_id"] != PAPER_ID:
        raise ValueError("paper identity mismatch")
    if value["upstream_revision"] != UPSTREAM_REVISION:
        raise ValueError("paper revision mismatch")
    return value


def load_claim_bindings(path: Path) -> tuple[ClaimBinding, ...]:
    record = load_source_record(path)
    bindings = []
    for item in record["claims"]:
        bindings.append(
            ClaimBinding(
                challenge_claim=item["challenge_claim"],
                challenge_claim_sha256=item["challenge_claim_sha256"],
                target_claim=item.get("target_claim"),
                targeted=item["targeted"],
                equations=tuple(item.get("equations", ())),
            )
        )
    return tuple(bindings)
