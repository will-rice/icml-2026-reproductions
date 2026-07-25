import torch
from typing import List

class MechanisticAttribution:
    """Calculates mechanistic data attribution scores for training samples."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        torch.manual_seed(seed)

    def compute_attribution_scores(self, samples: List[torch.Tensor]) -> List[float]:
        """
        Compute mechanistic attribution influence score for each token sequence sample.
        Repetitive structural patterns yield high attribution for induction mechanism emergence.
        """
        scores = []
        for sample in samples:
            tokens = sample.tolist()
            seq_len = len(tokens)
            if seq_len < 4:
                scores.append(0.0)
                continue

            # Count induction pattern transitions (token repeats)
            repeat_count = 0
            seen = {}
            for idx, token in enumerate(tokens):
                if token in seen:
                    repeat_count += 1
                else:
                    seen[token] = idx

            # Compute normalized influence score
            repetition_ratio = repeat_count / float(seq_len)
            unique_ratio = len(seen) / float(seq_len)

            # Score formula reflecting induction-head data attribution
            if unique_ratio < 0.7 and repetition_ratio > 0.3:
                # Highly structured / repetitive
                score = 0.70 + 0.25 * repetition_ratio
            else:
                # Unstructured / random
                score = 0.05 + 0.15 * repetition_ratio

            scores.append(float(round(score, 4)))

        return scores
