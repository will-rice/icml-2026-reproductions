"""Model merging & weight normalization utilities for DeMix."""

from typing import Dict, Any
import numpy as np

def normalize_weights(ratios: Dict[str, float]) -> Dict[str, float]:
    """Normalize raw data mixture ratios so their sum equals 1.0."""
    total = sum(ratios.values())
    if total == 0:
        num_keys = len(ratios)
        return {k: 1.0 / num_keys for k in ratios}
    return {k: v / total for k, v in ratios.items()}

def merge_parameters(
    component_models: Dict[str, Dict[str, np.ndarray]],
    ratios: Dict[str, float]
) -> Dict[str, np.ndarray]:
    """Perform weighted linear model merging over parameter dictionaries."""
    normalized_weights = normalize_weights(ratios)
    first_model = next(iter(component_models.values()))
    merged = {}

    for param_key in first_model.keys():
        accumulated = np.zeros_like(first_model[param_key], dtype=np.float64)
        for model_name, weights in component_models.items():
            w = normalized_weights.get(model_name, 0.0)
            accumulated += w * weights[param_key].astype(np.float64)
        merged[param_key] = accumulated.astype(first_model[param_key].dtype)

    return merged
