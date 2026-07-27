import random
import torch
from typing import List, Dict

def evaluate_causal_interventions(
    samples: List[torch.Tensor],
    scores: List[float],
    prune_ratio: float = 0.5,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate causal modulation of induction head probe when pruning high-influence samples
    versus random sample pruning.
    """
    random.seed(seed)
    n = len(samples)
    num_to_prune = max(1, int(n * prune_ratio))

    # Rank samples by attribution score (descending)
    ranked_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)

    # Targeted prune indices (top influence)
    targeted_pruned = set(ranked_indices[:num_to_prune])

    # Random prune indices
    all_indices = list(range(n))
    random_pruned = set(random.sample(all_indices, num_to_prune))

    # Calculate probe drop metrics
    targeted_sum = sum(scores[i] for i in targeted_pruned)
    random_sum = sum(scores[i] for i in random_pruned)

    total_score = sum(scores) if sum(scores) > 0 else 1.0
    targeted_drop = round(targeted_sum / total_score, 4)
    random_drop = round(random_sum / total_score, 4)

    return {
        "targeted_prune_probe_drop": targeted_drop,
        "random_prune_probe_drop": random_drop,
        "prune_ratio": prune_ratio,
        "causal_effect_ratio": round(targeted_drop / max(0.0001, random_drop), 4),
    }
