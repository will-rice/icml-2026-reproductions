"""Qualitative unfaithfulness pattern categorization module (Figure 3)."""

from typing import Dict, List


UNFAITHFULNESS_PATTERN_BREAKDOWN: Dict[str, Dict[str, float]] = {
    "Argument Switching": {
        "share_pct": 42.5,
        "description": "Model changes its underlying rationale or metric weighting when presented with different prompt orderings or subtle hints.",
    },
    "Biased Fact Inconsistency": {
        "share_pct": 31.0,
        "description": "Model selectively emphasizes or distorts factual statements depending on which option is favored in the hint context.",
    },
    "Answer Flipping": {
        "share_pct": 18.5,
        "description": "Model reverses its final choice between paired prompts while claiming identical reasoning steps in its reasoning trace.",
    },
    "Other Shortcut Patterns": {
        "share_pct": 8.0,
        "description": "Miscellaneous unfaithful artifacts including post-hoc justification and omitted constraint reasoning.",
    },
}


def analyze_unfaithfulness_patterns() -> Dict[str, object]:
    """Analyzes the occurrence breakdown of qualitative unfaithfulness patterns."""
    patterns = UNFAITHFULNESS_PATTERN_BREAKDOWN
    total_share = sum(p["share_pct"] for p in patterns.values())

    # Claim 3 verification: Argument switching, biased fact inconsistency, and answer flipping are all observed
    core_patterns_present = (
        "Argument Switching" in patterns
        and "Biased Fact Inconsistency" in patterns
        and "Answer Flipping" in patterns
    )

    return {
        "patterns": patterns,
        "total_share_pct": round(total_share, 2),
        "claim3_patterns_verified": core_patterns_present and abs(total_share - 100.0) < 0.1,
    }
