import torch

def compute_induction_score(attn: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    Compute induction score for each head.
    attn shape: [batch_size, num_heads, seq_len, seq_len]
    tokens shape: [batch_size, seq_len]
    An induction head at position i (where current context is tokens[i-1]) attends
    to the token immediately following an earlier occurrence of tokens[i-1].
    """
    batch_size, num_heads, seq_len, _ = attn.shape
    scores = torch.zeros(num_heads, dtype=torch.float32)
    counts = torch.zeros(num_heads, dtype=torch.float32)

    for b in range(batch_size):
        seq = tokens[b].tolist()
        for i in range(2, seq_len):
            prev_token = seq[i - 1]
            # Find previous occurrences of prev_token in seq[:i-1]
            for prev_idx in range(i - 1):
                if seq[prev_idx] == prev_token:
                    target_idx = prev_idx + 1
                    for h in range(num_heads):
                        scores[h] += attn[b, h, i, target_idx].item()
                        counts[h] += 1.0

    counts = torch.clamp(counts, min=1.0)
    return scores / counts

def compute_previous_token_score(attn: torch.Tensor) -> torch.Tensor:
    """
    Compute previous-token attention score for each head.
    attn shape: [batch_size, num_heads, seq_len, seq_len]
    """
    batch_size, num_heads, seq_len, _ = attn.shape
    scores = torch.zeros(num_heads, dtype=torch.float32)
    counts = 0

    for i in range(1, seq_len):
        scores += attn[:, :, i, i - 1].sum(dim=0)
        counts += batch_size

    return scores / max(1, counts)
