import torch
from typing import List, Dict

def classify_sample_pattern(sample: torch.Tensor) -> str:
    tokens = sample.tolist()
    seq_len = len(tokens)
    unique_tokens = set(tokens)
    if len(unique_tokens) < seq_len * 0.7:
        return "repetitive_structural"
    return "unstructured_random"

def analyze_pattern_attribution(samples: List[torch.Tensor], scores: List[float]) -> Dict[str, float]:
    rep_scores = []
    rand_scores = []
    
    for sample, score in zip(samples, scores):
        category = classify_sample_pattern(sample)
        if category == "repetitive_structural":
            rep_scores.append(score)
        else:
            rand_scores.append(score)

    mean_rep = sum(rep_scores) / max(1, len(rep_scores)) if rep_scores else 0.0
    mean_rand = sum(rand_scores) / max(1, len(rand_scores)) if rand_scores else 0.0

    return {
        "mean_repetitive_score": round(mean_rep, 4),
        "mean_unstructured_score": round(mean_rand, 4),
        "repetitive_count": len(rep_scores),
        "unstructured_count": len(rand_scores),
    }
