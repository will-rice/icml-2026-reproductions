"""Layer-range profiler and token selector for RelayCaching."""

import numpy as np


class LayerRangeProfiler:
    """Profiles KV cache similarity across Transformer layers to identify critical rectification ranges."""

    def __init__(self, num_layers: int = 32):
        self.num_layers = num_layers

    def compute_layer_similarities(
        self, decoding_kv: np.ndarray, prefill_kv: np.ndarray
    ) -> np.ndarray:
        """Computes cosine similarity per layer between decoding and prefill KV caches.

        Returns an array of shape (num_layers,) showing the U-shaped alignment curve.
        """
        similarities = []
        for l in range(self.num_layers):
            d_l = decoding_kv[l].flatten()
            p_l = prefill_kv[l].flatten()
            norm_d = np.linalg.norm(d_l)
            norm_p = np.linalg.norm(p_l)
            if norm_d == 0 or norm_p == 0:
                sim = 1.0
            else:
                sim = float(np.dot(d_l, p_l) / (norm_d * norm_p))
            similarities.append(sim)
        return np.array(similarities)

    def identify_critical_layers(
        self, similarities: np.ndarray, threshold: float = 0.92
    ) -> list[int]:
        """Identifies layers where similarity falls below threshold requiring rectification."""
        return [int(l) for l, sim in enumerate(similarities) if sim < threshold]


class TokenSelector:
    """Selects specific tokens requiring rectification via deviation and influence scoring."""

    def __init__(self, deviation_weight: float = 0.6, influence_weight: float = 0.4):
        self.deviation_weight = deviation_weight
        self.influence_weight = influence_weight

    def select_tokens_for_rectification(
        self,
        decoding_layer_kv: np.ndarray,
        prefill_layer_kv: np.ndarray,
        reuse_target_ratio: float = 0.85,
    ) -> list[int]:
        """Selects token indices to recompute to achieve desired cache reuse target."""
        seq_len = decoding_layer_kv.shape[0]
        rectify_count = int(np.ceil(seq_len * (1.0 - reuse_target_ratio)))

        # Deviation per token
        diffs = np.linalg.norm(decoding_layer_kv - prefill_layer_kv, axis=-1)
        # Influence score (tokens closer to end of context have higher attention influence)
        positions = np.arange(seq_len)
        influence = (positions + 1) / seq_len

        combined_scores = self.deviation_weight * diffs + self.influence_weight * influence
        selected_indices = np.argsort(combined_scores)[::-1][:rectify_count]
        return sorted([int(i) for i in selected_indices])
