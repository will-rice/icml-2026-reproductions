"""Pinned claim identity catalog for schema version 2 evidence outputs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    id: str
    digest: str
    text: str
    section: str
    kind: str


LIVE_CLAIM_TEXTS = (
    "Theorem 1 bounds the total variation distance between the DDPM output and target distribution as TV(X0,Y1) ≲ log²(KT)log²T/T + ε_score√log T + √(dε_approx)log^(3/2)(dT), yielding Õ(1/ε) discretization steps for ε-accurate sampling independent of ambient dimension d and mixture count K (Theorem 1).",
    "The dimension-free discretization rate relies on Assumption 1, requiring the target distribution to be ε_approx-close in TV distance to an isotropic Gaussian mixture whose component means satisfy the polynomial growth bound ||μ_k||_2 ≤ T^{c_r} (Assumption 1).",
    "The bound remains valid under imperfect score estimation via Assumption 2, which requires the time-averaged squared L2 score error to satisfy (1/T)∑ε²_score,t ≤ ε²_score, contributing an additive ε_score√log T term to the TV bound (Assumption 2, Theorem 1).",
    "The proof's key technical lemma shows the trace of the score Jacobian for a Gaussian mixture satisfies tr(I_d + J_t(x)) ≤ C1 log(KT) with high probability, a bound with no explicit dependence on ambient dimension d, enabling the dimension-free discretization analysis.",
    "The paper contrasts its Õ(1/ε)-iteration, dimension-free rate against prior DDPM convergence results (e.g., Li & Yan 2024; Liang et al. 2024) that required O(d/ε) iterations even when the target is a Gaussian mixture, showing the mixture structure can be exploited to remove the dimension dependence entirely (comparison to prior work).",
)


LIVE_CLAIMS = (
    Claim(
        id="theorem-1-dimension-free-rate",
        digest=hashlib.sha256(LIVE_CLAIM_TEXTS[0].encode("utf-8")).hexdigest(),
        text=LIVE_CLAIM_TEXTS[0],
        section="Theorem 1",
        kind="theorem",
    ),
    Claim(
        id="assumption-1-mixture-structure",
        digest=hashlib.sha256(LIVE_CLAIM_TEXTS[1].encode("utf-8")).hexdigest(),
        text=LIVE_CLAIM_TEXTS[1],
        section="Assumption 1",
        kind="assumption",
    ),
    Claim(
        id="assumption-2-score-error",
        digest=hashlib.sha256(LIVE_CLAIM_TEXTS[2].encode("utf-8")).hexdigest(),
        text=LIVE_CLAIM_TEXTS[2],
        section="Assumption 2",
        kind="assumption",
    ),
    Claim(
        id="lemma-1-jacobian-trace-bound",
        digest=hashlib.sha256(LIVE_CLAIM_TEXTS[3].encode("utf-8")).hexdigest(),
        text=LIVE_CLAIM_TEXTS[3],
        section="Lemma 1",
        kind="lemma",
    ),
    Claim(
        id="comparison-prior-work",
        digest=hashlib.sha256(LIVE_CLAIM_TEXTS[4].encode("utf-8")).hexdigest(),
        text=LIVE_CLAIM_TEXTS[4],
        section="Comparison to prior work",
        kind="comparison",
    ),
)


def validate_claim_text(text_or_digest: str) -> Claim:
    """Return the pinned Claim matching text, digest, or id, or raise ValueError."""
    for claim in LIVE_CLAIMS:
        if text_or_digest in (claim.text, claim.digest, claim.id):
            return claim
    raise ValueError(f"invalid claim text or digest: {text_or_digest}")
