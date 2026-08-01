"""
Test-Time Training (TTT) Streaming Memory Layer Implementation for video-SALMONN S.
Pure Python implementation with zero required third-party dependencies.
"""

from typing import Dict, Any, List, Tuple
import math

class TTTStreamingMemoryLayer:
    """
    TTT Streaming Memory Layer that uses fast-weight updates
    as streaming memory for long-sequence audio-visual inputs.
    """
    def __init__(self, hidden_dim: int, memory_dim: int, learning_rate: float = 0.01):
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.learning_rate = learning_rate

        # Base projection matrices
        self.W_key = [[0.01 * (i + j) for j in range(memory_dim)] for i in range(hidden_dim)]
        self.W_val = [[0.01 * (i - j) for j in range(memory_dim)] for i in range(hidden_dim)]

        # Fast weights identity matrix W_fast: shape (memory_dim, memory_dim)
        self.reset_fast_weights()
        self.ttt_frozen = False

    def set_freeze_ttt(self, freeze: bool):
        """Freeze or unfreeze TTT base parameters (Stage 2 training control)."""
        self.ttt_frozen = freeze

    def reset_fast_weights(self):
        """Reset fast-weight streaming memory state."""
        self.W_fast = [[1.0 if i == j else 0.0 for j in range(self.memory_dim)] for i in range(self.memory_dim)]

    def _matmul_vec(self, vec: List[float], mat: List[List[float]]) -> List[float]:
        """Vector-matrix multiplication."""
        out = [0.0] * len(mat[0])
        for j in range(len(mat[0])):
            out[j] = sum(vec[i] * mat[i][j] for i in range(len(vec)))
        return out

    def fast_weight_update(self, k: List[float], v: List[float]) -> float:
        """
        Perform fast-weight update step: W_fast = W_fast - lr * grad
        Returns step MSE loss.
        """
        pred = self._matmul_vec(k, self.W_fast)
        err = [p - val for p, val in zip(pred, v)]
        loss = sum(e ** 2 for e in err) / max(1, len(err))

        if not self.ttt_frozen:
            # Gradient update step
            for i in range(self.memory_dim):
                for j in range(self.memory_dim):
                    self.W_fast[i][j] -= self.learning_rate * err[i] * k[j]
        return loss

    def forward(self, sequence: List[List[float]]) -> Tuple[List[List[float]], float]:
        """
        Forward pass over input sequence of shape (seq_len, hidden_dim).
        Returns predicted memory outputs and average prediction loss.
        """
        total_loss = 0.0
        outputs = []
        for x in sequence:
            k = self._matmul_vec(x, self.W_key)
            v = self._matmul_vec(x, self.W_val)
            pred = self._matmul_vec(k, self.W_fast)
            outputs.append(pred)
            step_loss = self.fast_weight_update(k, v)
            total_loss += step_loss

        avg_loss = total_loss / max(1, len(sequence))
        return outputs, avg_loss


def compute_memory_token_reduction(
    seq_len: int,
    memory_dim: int,
    similarity_merge_ratio: float = 0.5
) -> Dict[str, Any]:
    """
    Compute memory token footprint comparison between TTT streaming memory
    and similarity-based token merging over a long video stream.
    """
    ttt_token_footprint = memory_dim
    similarity_token_footprint = int(seq_len * similarity_merge_ratio)
    ratio = ttt_token_footprint / max(1, similarity_token_footprint)

    return {
        "sequence_length": seq_len,
        "ttt_tokens": ttt_token_footprint,
        "similarity_merge_tokens": similarity_token_footprint,
        "ratio_ttt_to_similarity": ratio,
        "achieves_under_25_percent": ratio < 0.25
    }
