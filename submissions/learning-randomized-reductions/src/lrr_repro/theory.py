"""Theory audit module for finite correlated-sampling models and paper definitions."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping


@dataclass(frozen=True)
class FiniteRSR:
    domain: tuple[int, ...]
    randomness: tuple[int, ...]
    queries: tuple[Callable[[int, int], int], ...]
    recovery: Callable[[int, int, tuple[int, ...]], int]


@dataclass(frozen=True)
class TheoryAudit:
    marginal_uniform: bool
    perfect_for_hypothesis: bool
    epsilon: Fraction
    good_input_fraction: Fraction
    minimum_recovery_probability: Fraction
    implication_holds: bool


def modular_addition_rsr(n: int = 4) -> FiniteRSR:
    domain = tuple(range(n))
    randomness = tuple(range(n))
    queries = (
        lambda x, r: (x + r) % n,
        lambda x, r: r,
    )
    recovery = lambda x, r, y: (y[0] - y[1]) % n
    return FiniteRSR(domain=domain, randomness=randomness, queries=queries, recovery=recovery)


def nonuniform_rsr() -> FiniteRSR:
    domain = (0, 1, 2, 3)
    randomness = (0, 1, 2, 3)
    queries = (
        lambda x, r: 0,
        lambda x, r: r,
    )
    recovery = lambda x, r, y: y[1]
    return FiniteRSR(domain=domain, randomness=randomness, queries=queries, recovery=recovery)


def audit_claim_a1(
    rsr: FiniteRSR,
    truth: Mapping[int, int],
    hypothesis: Mapping[int, int],
    rho: Fraction,
    xi: Fraction,
) -> TheoryAudit:
    num_domain = len(rsr.domain)
    num_rand = len(rsr.randomness)
    expected_freq = Fraction(num_rand, num_domain)

    # Verify marginal uniformity for each query on each input
    for q_idx, q in enumerate(rsr.queries):
        for x in rsr.domain:
            counts: dict[int, int] = {}
            for r in rsr.randomness:
                out = q(x, r)
                counts[out] = counts.get(out, 0) + 1
            for d in rsr.domain:
                if Fraction(counts.get(d, 0), 1) != expected_freq:
                    raise ValueError(
                        f"marginal uniformity failed for query {q_idx} on input {x}"
                    )

    # Verify perfect recovery for hypothesis
    perfect_for_hyp = True
    for x in rsr.domain:
        for r in rsr.randomness:
            queries_out = tuple(q(x, r) for q in rsr.queries)
            hyp_queries_out = tuple(hypothesis[y] for y in queries_out)
            rec = rsr.recovery(x, r, hyp_queries_out)
            if rec != hypothesis[x]:
                perfect_for_hyp = False
                break

    # Epsilon = fraction of inputs where hypothesis differs from truth
    wrong_count = sum(1 for x in rsr.domain if hypothesis[x] != truth[x])
    epsilon = Fraction(wrong_count, num_domain)

    # Compute recovery probability for each input x
    good_inputs = 0
    rec_probs = []
    threshold = 1 - rho

    for x in rsr.domain:
        correct_r = 0
        for r in rsr.randomness:
            queries_out = tuple(q(x, r) for q in rsr.queries)
            hyp_queries_out = tuple(hypothesis[y] for y in queries_out)
            rec = rsr.recovery(x, r, hyp_queries_out)
            if rec == truth[x]:
                correct_r += 1
        prob = Fraction(correct_r, num_rand)
        rec_probs.append(prob)
        if prob >= threshold:
            good_inputs += 1

    good_input_fraction = Fraction(good_inputs, num_domain)
    good_probs = [prob for prob in rec_probs if prob >= threshold]
    min_rec_prob = min(good_probs) if good_probs else Fraction(0, 1)

    implication_holds = good_input_fraction >= (1 - xi)

    return TheoryAudit(
        marginal_uniform=True,
        perfect_for_hypothesis=perfect_for_hyp,
        epsilon=epsilon,
        good_input_fraction=good_input_fraction,
        minimum_recovery_probability=min_rec_prob,
        implication_holds=implication_holds,
    )


def extract_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise ValueError(f"PDF file missing at {pdf_path}")
    import pypdf

    reader = pypdf.PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in reader.pages)


def verify_theory_locators(
    context: Mapping[str, object], cache_dir: Path | None = None
) -> None:
    versions = context.get("versions", {})
    if not isinstance(versions, dict) or "v5" not in versions:
        raise ValueError("Missing version v5 in paper context")
    v5 = versions["v5"]
    defs = v5.get("definitions", {})
    claims = v5.get("claims", {})

    required_defs = ["def_4_1", "def_4_3", "def_4_5"]
    required_claims = ["claim_a1", "claim_a2"]

    for d in required_defs:
        if d not in defs:
            raise ValueError(f"Missing definition {d} in paper context v5")
    for c in required_claims:
        if c not in claims:
            raise ValueError(f"Missing claim {c} in paper context v5")

    if cache_dir is not None:
        pdf_v5 = cache_dir / "2412.18134v5.pdf"
        pdf_text = extract_pdf_text(pdf_v5)
        pdf_text_clean = " ".join(pdf_text.split())

        # Extract & verify definitions and claims directly from v5 PDF text
        expected_patterns = {
            "def_4_1": ["Definition 4.1", "randomized self-reduction"],
            "def_4_3": ["Definition 4.3", "sample access"],
            "def_4_5": ["Definition 4.5", "learning from correlated"],
            "claim_a1": ["Claim", "A.1"],
            "claim_a2": ["Claim", "A.2"],
        }

        for item_key, patterns in expected_patterns.items():
            for pat in patterns:
                if pat.lower() not in pdf_text_clean.lower():
                    raise ValueError(
                        f"PDF text missing required pattern '{pat}' for {item_key}"
                    )

        # Ensure context definition names match extracted PDF text
        for d in required_defs:
            name = defs[d].get("name", "")
            # Check if all key words from definition name occur in PDF text
            words = [w.strip("(),.").lower() for w in name.split() if len(w.strip("(),.")) > 3]
            matched = all(w in pdf_text_clean.lower() for w in words)
            if not matched:
                raise ValueError(
                    f"PDF text missing matching terms for definition name '{name}'"
                )
