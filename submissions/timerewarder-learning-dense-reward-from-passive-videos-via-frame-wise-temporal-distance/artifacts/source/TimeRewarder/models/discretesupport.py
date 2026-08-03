import torch
import numpy as np


def transform_one(x):
    return np.sign(x) * (np.sqrt(np.abs(x) + 1.0) - 1) + 0.001 * x

class DiscreteSupport(object):
    def __init__(self, **kwargs):
        if 'range' in kwargs:
            self.range = kwargs['range']
        else:
            self.range = (-1, 1)
        if 'bins' in kwargs:
            self.bins = kwargs['bins']
        else:
            self.bins = 20
        if 'softmax_temp' in kwargs:
            self.softmax_temp = kwargs['softmax_temp']
        else:
            self.softmax_temp = 1.0

    def scalar_to_vector(self, x):
        """ Reference from MuZero: Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        x_min = self.range[0]
        x_max = self.range[1]
        bins = self.bins

        epsilon = 0.001

        x_min = transform_one(x_min)
        x_max = transform_one(x_max)
        
        scale = (x_max - x_min) / (bins - 1)

        sign = torch.ones(x.shape).float().to(x.device)
        sign[x < 0] = -1.0

        x = sign * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x
        x = x / scale

        x.clamp_(x_min / scale, x_max / scale - 1e-5)
        x = x - x_min / scale
        x_low_idx = x.floor()
        x_high_idx = x.ceil()
        p_high = x - x_low_idx
        p_low = 1 - p_high

        target = torch.zeros(tuple(x.shape) + (bins,), dtype=p_high.dtype).to(x.device)
        target.scatter_(len(x.shape), x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
        target.scatter_(len(x.shape), x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))

        return target
    
    def vector_to_scalar(self, logits):
        """ Inverse operation of scalar_to_vector to convert probability distribution back to scalar value
        """
        x_min = self.range[0]
        x_max = self.range[1]
        bins = self.bins
        softmax_temp = self.softmax_temp

        epsilon = 0.001

        x_min = transform_one(x_min)
        x_max = transform_one(x_max)
        scale = (x_max - x_min) / (bins - 1)
        x_range = np.linspace(x_min, x_max, bins)

        # Convert logits to probabilities and compute expected value
        logits = logits.to(dtype=torch.float64)  # Convert to float64 (double) for higher precision
        probs = torch.softmax(logits / softmax_temp, dim=-1)  # softmax_temp<1 sharpens / >1 flattens (default 1.0 = unchanged)
        support = torch.tensor(x_range, device=logits.device).expand(probs.shape)
        value = (support * probs).sum(-1, keepdim=True)

        # Inverse transform_one operation
        sign = torch.sign(value)
        abs_value = ((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1
        value = sign * abs_value

        return value


LEGACY_ORDER_CLS_DIM = 2

_ORDER_HEAD_KEYS = (
    ('order_mlp.1.weight', 'order_mlp.1.bias'),
    ('order_linear.weight', 'order_linear.bias'),
)


def order_head_regression_dim(bin_num, use_bin=True):
    if use_bin and bin_num > 0:
        return int(bin_num)
    return 1


def strip_legacy_order_head_state_dict(state_dict, regression_dim):
    """Drop legacy wrong-text cls dims when loading into a regression-only head."""
    for w_key, b_key in _ORDER_HEAD_KEYS:
        if w_key not in state_dict:
            continue
        w = state_dict[w_key]
        if w.shape[0] == regression_dim + LEGACY_ORDER_CLS_DIM:
            state_dict[w_key] = w[LEGACY_ORDER_CLS_DIM:].clone()
            if b_key in state_dict:
                state_dict[b_key] = state_dict[b_key][LEGACY_ORDER_CLS_DIM:].clone()
    return state_dict


def strip_legacy_order_head_checkpoint(checkpoint, regression_dim):
    """Strip unused legacy cls output dims from a training/export checkpoint dict."""
    if not isinstance(checkpoint, dict):
        return checkpoint
    if 'model' in checkpoint:
        checkpoint['model'] = strip_legacy_order_head_state_dict(
            checkpoint['model'], regression_dim)
    return checkpoint


def slice_regression_logits(logits, regression_dim):
    if logits.shape[-1] == regression_dim:
        return logits
    if logits.shape[-1] == regression_dim + LEGACY_ORDER_CLS_DIM:
        return logits[..., LEGACY_ORDER_CLS_DIM:]
    raise ValueError(
        f'unexpected order head dim {logits.shape[-1]} (expected {regression_dim} or '
        f'{regression_dim + LEGACY_ORDER_CLS_DIM})'
    )