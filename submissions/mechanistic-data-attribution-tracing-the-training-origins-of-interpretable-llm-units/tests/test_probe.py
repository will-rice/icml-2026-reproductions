import pytest
import torch
from mechanistic_data_attribution_repro.probe import compute_induction_score, compute_previous_token_score

def test_compute_induction_score():
    # Sequence with repeated tokens: [A B C A B C] -> token 3 (A) should predict token 4 (B)
    tokens = torch.tensor([[10, 20, 30, 10, 20, 30]])
    # Mock attention matrix shape [batch, heads, seq, seq]
    batch_size, num_heads, seq_len = 1, 4, 6
    attn = torch.zeros(batch_size, num_heads, seq_len, seq_len)
    # Head 0 has strong induction pattern: pos 4 attends to pos 1 (previous token after first occurrence of 10)
    attn[0, 0, 4, 1] = 0.9
    attn[0, 0, 5, 2] = 0.8

    scores = compute_induction_score(attn, tokens)
    assert scores.shape == (num_heads,)
    assert scores[0].item() > 0.5
    assert scores[1].item() < 0.2

def test_compute_previous_token_score():
    batch_size, num_heads, seq_len = 1, 2, 5
    attn = torch.zeros(batch_size, num_heads, seq_len, seq_len)
    # Head 0 attends to previous token (i, i-1)
    for i in range(1, seq_len):
        attn[0, 0, i, i - 1] = 0.95

    scores = compute_previous_token_score(attn)
    assert scores.shape == (num_heads,)
    assert scores[0].item() > 0.8
