#!/usr/bin/env python3
"""
sampling_strategies.py  --  TD3B / MDLM validity-boosting samplers (FUNCTION B).

Goal
----
As the target peptide length grows, the fraction of decoded SMILES that pass
``utils.app.PeptideAnalyzer.is_peptide`` (RDKit) collapses.  This module supplies a
library of *sampling-time* techniques (no retraining) that raise that valid-yield,
especially at long length.

All samplers REUSE the model's existing masked-diffusion primitives:
    * ``Diffusion.sample_prior``        -> fully masked start
    * ``Diffusion.single_reverse_step`` -> one reverse (denoising) step  (returns log_p, x_next)
    * ``Diffusion.single_noise_removal``-> final step guaranteeing no surviving [MASK]
    * ``Diffusion.forward``             -> per-position log p(x0)  (SUBS-parameterised logits)

We DO NOT reimplement the diffusion math.  The reverse-posterior construction
(``q_xs = p_x0 * (t - (t-dt))`` with the mask-stay probability, and the carry-over
copy_flag) is left entirely to ``single_reverse_step`` / ``single_noise_removal``.
We only change **token SELECTION** -- i.e. we transform the model's clean-token
distribution ``p_x0`` (temperature / top-k / top-p) that we feed back into
``single_reverse_step(..., p_x0=...)`` -- and add a **remask self-correction loop**
and a **best-of-N validity-guided rejection** wrapper on top.

Common entry point
------------------
    generate(model, tokenizer, analyzer, batch_size, length, strategy="baseline", **kw)
        -> (tokens, sequences, valid_mask, stats)

Strategies (``strategy=``):
    baseline    : reverse diffusion identical to inference.sample_sequences (num_steps=128)
    more_steps  : same, but num_steps scales with length (steps_per_token)
    top_p       : nucleus -- restrict p_x0 to the smallest set with cumulative mass >= p
    nucleus     : alias of top_p
    top_k       : restrict p_x0 to the k most-probable clean tokens
    low_temp    : temperature < 1 on the clean-token logits (sharper, more valid)
    remask      : self-correction -- generate, then remask the lowest-confidence K%%
                  of tokens of INVALID sequences and re-denoise, for R rounds
    best_of_n   : oversample N per slot, keep the first valid decode
    nucleus_remask : nucleus + remask (recommended default for long length)

Any preset can be combined with explicit kwargs, e.g.
    generate(..., strategy="remask", top_p=0.9, remask_rounds=4)

The library is model-agnostic: it works with a RANDOM-init ``Diffusion`` (for CPU
development / relative benchmarking) and, unchanged, with the real checkpoint.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------- #
#  Distribution transforms  (token SELECTION only -- no diffusion math here)   #
# --------------------------------------------------------------------------- #
_NEG_INF = -1e9


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest logits per position, push the rest to -inf."""
    if k is None or k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]  # (..., 1) k-th largest
    return logits.masked_fill(logits < kth, _NEG_INF)


def _apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filter: zero out the low-mass tail so the kept set has cumulative
    mass >= p (always keeps at least the argmax), then renormalise."""
    if p is None or p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    # a token is DROPPED if the mass strictly *before* it already reached p
    drop_sorted = (cumsum - sorted_probs) > p
    sorted_probs = sorted_probs.masked_fill(drop_sorted, 0.0)
    new_probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    return new_probs / new_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def make_transform(temperature: float = 1.0,
                   top_p: Optional[float] = None,
                   top_k: Optional[int] = None) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Build a callable ``log_p (B,L,V) -> p_x0 (B,L,V)`` implementing
    temperature -> top_k (logit space) -> softmax -> top_p (prob space).

    Returns ``None`` when the transform is the identity (baseline path), so the
    caller can take the cheaper single-``single_reverse_step`` route.
    """
    temp_on = temperature is not None and abs(temperature - 1.0) > 1e-8
    if not temp_on and top_p is None and top_k is None:
        return None

    def _transform(log_p: torch.Tensor) -> torch.Tensor:
        v = log_p
        if temp_on:
            v = v / float(temperature)
        if top_k is not None:
            v = _apply_top_k(v, int(top_k))
        probs = torch.softmax(v, dim=-1)
        if top_p is not None:
            probs = _apply_top_p(probs, float(top_p))
        return probs

    return _transform


# --------------------------------------------------------------------------- #
#  Core reverse-diffusion loop (delegates all diffusion math to the model)     #
# --------------------------------------------------------------------------- #
def _reverse_step(model, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor,
                  attn_mask: torch.Tensor,
                  transform: Optional[Callable]) -> Tuple[torch.Tensor, torch.Tensor]:
    """One reverse step.  Returns (log_p, x_next).

    * transform is None  -> plain ``single_reverse_step`` (one forward, returns log_p).
    * transform given    -> one ``forward`` to get log p_x0, transform it, then hand the
      modified p_x0 to ``single_reverse_step`` which still builds the reverse posterior
      and samples exactly as in the original math.
    """
    if transform is None:
        log_p, x_next = model.single_reverse_step(x, t=t, dt=dt, attn_mask=attn_mask)
        return log_p, x_next
    # lower-level logits (SUBS-parameterised); sigma comes from the model's own schedule
    sigma_t, _ = model.noise(t)
    log_p = model.forward(x, attn_mask=attn_mask, sigma=sigma_t)
    p_x0 = transform(log_p)
    _, x_next = model.single_reverse_step(x, t=t, dt=dt, p_x0=p_x0, attn_mask=attn_mask)
    return log_p, x_next


def _update_confidence(conf: torch.Tensor, x_prev: torch.Tensor, x_next: torch.Tensor,
                       log_p: torch.Tensor, mask_index: int) -> torch.Tensor:
    """Record, for every position that transitions mask->token at this step, the model's
    probability of the chosen token.  Because of carry-over unmasking each position is
    written exactly once (when it first unmasks); higher conf == model more certain."""
    if log_p is None:
        return conf
    newly = (x_prev == mask_index) & (x_next != mask_index)
    if not newly.any():
        return conf
    probs = log_p.exp()
    chosen = x_next.clamp(0, probs.shape[-1] - 1).unsqueeze(-1)
    step_conf = probs.gather(-1, chosen).squeeze(-1)
    return torch.where(newly, step_conf, conf)


def _reverse_diffusion(model, x: torch.Tensor, num_steps: int,
                       t_start: float = 1.0, eps: float = 1e-5,
                       transform: Optional[Callable] = None,
                       attn_mask: Optional[torch.Tensor] = None,
                       conf: Optional[torch.Tensor] = None,
                       noise_removal: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reverse-diffuse ``x`` from time ``t_start`` down to ``eps`` over ``num_steps``.

    Mirrors ``inference.sample_sequences`` (linspace timesteps, final noise-removal to
    guarantee no surviving [MASK]) but supports (a) an arbitrary starting time -- so the
    remask loop can resume from a partially-masked state -- and (b) a p_x0 ``transform``.
    Already-unmasked positions are preserved by the model's copy_flag (carry-over).
    Returns (x, confidence) with confidence in [0, 1] per position.
    """
    device = model.device
    x = x.to(device, dtype=torch.long)
    if attn_mask is None:
        attn_mask = torch.ones_like(x, device=device, dtype=torch.long)
    if conf is None:
        conf = torch.zeros(x.shape, device=device, dtype=torch.float32)
    mask_index = model.mask_index

    num_steps = max(int(num_steps), 1)
    timesteps = torch.linspace(t_start, eps, num_steps + 1, device=device)
    dt = torch.tensor((t_start - eps) / num_steps, device=device)

    for i in range(num_steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
        log_p, x_next = _reverse_step(model, x, t, dt, attn_mask, transform)
        conf = _update_confidence(conf, x, x_next, log_p, mask_index)
        x = x_next.to(device)

    if noise_removal and (x == mask_index).any():
        t = timesteps[-2] * torch.ones(x.shape[0], 1, device=device)
        log_p, x_next = model.single_noise_removal(x, t=t, dt=dt, attn_mask=attn_mask)
        conf = _update_confidence(conf, x, x_next, log_p, mask_index)
        x = x_next.to(device)

    return x, conf


# --------------------------------------------------------------------------- #
#  Decode / validity helpers                                                    #
# --------------------------------------------------------------------------- #
def decode_and_validate(tokenizer, analyzer, x: torch.Tensor) -> Tuple[List[str], np.ndarray]:
    """Decode token ids to SMILES and test each with ``analyzer.is_peptide``."""
    sequences = tokenizer.batch_decode(x)
    valid = np.fromiter((bool(analyzer.is_peptide(s)) for s in sequences),
                        dtype=bool, count=len(sequences))
    return sequences, valid


# --------------------------------------------------------------------------- #
#  Remask self-correction (the key technique for long sequences)               #
# --------------------------------------------------------------------------- #
def _remask_and_redenoise(model, x: torch.Tensor, conf: torch.Tensor,
                          rows: torch.Tensor, remask_frac: float, remask_steps: int,
                          transform: Optional[Callable], eps: float,
                          attn_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """For the selected ``rows`` (typically the invalid sequences): set the lowest-
    confidence ``remask_frac`` of their positions back to [MASK], then re-denoise ONLY
    those rows from t=remask_frac down to eps.  Returns updated (x, conf) copies.

    The high-confidence tokens are kept (carry-over), so re-denoising only re-samples
    the positions the model was least sure about -- a cheap, targeted second attempt.
    """
    x = x.clone()
    conf = conf.clone()
    sub_x = x[rows].clone()
    sub_conf = conf[rows].clone()
    n, L = sub_x.shape
    k = max(1, int(round(remask_frac * L)))
    k = min(k, L)

    # lowest-confidence k positions per row -> back to MASK
    low_idx = torch.topk(sub_conf, k, dim=-1, largest=False).indices  # (n, k)
    sub_x.scatter_(1, low_idx, model.mask_index)
    sub_conf.scatter_(1, low_idx, 0.0)  # will be recomputed when re-unmasked

    # resume masked-diffusion from a time consistent with the masked fraction (~k/L)
    t_start = float(min(max(k / L, 2 * eps), 1.0))
    sub_attn = attn_mask[rows]
    sub_x, sub_conf = _reverse_diffusion(
        model, sub_x, num_steps=remask_steps, t_start=t_start, eps=eps,
        transform=transform, attn_mask=sub_attn, conf=sub_conf, noise_removal=True)

    x[rows] = sub_x
    conf[rows] = sub_conf
    return x, conf


# --------------------------------------------------------------------------- #
#  Single generation pass (+ optional remask rounds)                           #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _generate_once(model, tokenizer, analyzer, batch_size: int, length: int,
                   num_steps: int, eps: float,
                   transform: Optional[Callable],
                   remask_rounds: int, remask_frac: float, remask_steps: int
                   ) -> Tuple[torch.Tensor, List[str], np.ndarray, torch.Tensor, List[int]]:
    device = model.device
    x = model.sample_prior(batch_size, length).to(device, dtype=torch.long)
    attn_mask = torch.ones_like(x, device=device, dtype=torch.long)

    x, conf = _reverse_diffusion(model, x, num_steps=num_steps, t_start=1.0, eps=eps,
                                 transform=transform, attn_mask=attn_mask,
                                 noise_removal=True)
    sequences, valid = decode_and_validate(tokenizer, analyzer, x)
    round_valid = [int(valid.sum())]  # valid count after each stage (round 0 == initial)

    for _ in range(int(remask_rounds)):
        if valid.all():
            break
        rows = torch.from_numpy(np.where(~valid)[0]).to(device=device, dtype=torch.long)
        x, conf = _remask_and_redenoise(model, x, conf, rows, remask_frac, remask_steps,
                                        transform, eps, attn_mask)
        # re-decode / re-validate only the rows we touched
        new_seqs = tokenizer.batch_decode(x[rows])
        for j, gi in enumerate(rows.tolist()):
            sequences[gi] = new_seqs[j]
            valid[gi] = bool(analyzer.is_peptide(new_seqs[j]))
        round_valid.append(int(valid.sum()))

    return x, sequences, valid, conf, round_valid


# --------------------------------------------------------------------------- #
#  Best-of-N validity-guided rejection wrapper                                 #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _generate_best_of_n(model, tokenizer, analyzer, batch_size: int, length: int,
                        num_steps: int, eps: float, transform: Optional[Callable],
                        remask_rounds: int, remask_frac: float, remask_steps: int,
                        best_of_n: int
                        ) -> Tuple[torch.Tensor, List[str], np.ndarray, Dict]:
    """Draw up to ``best_of_n`` independent candidates per slot; keep the first valid
    decode for each slot (fall back to the last draw if none is valid)."""
    device = model.device
    tokens = None
    sequences: List[str] = [""] * batch_size
    valid = np.zeros(batch_size, dtype=bool)
    draws_used = 0
    for n in range(max(int(best_of_n), 1)):
        draws_used = n + 1
        xg, seqs, vmask, _conf, _rv = _generate_once(
            model, tokenizer, analyzer, batch_size, length, num_steps, eps,
            transform, remask_rounds, remask_frac, remask_steps)
        if tokens is None:
            tokens = xg.clone()
            for i in range(batch_size):
                sequences[i] = seqs[i]
            valid = vmask.copy()
        else:
            # accept this draw only for slots not yet valid
            take = (~valid) & vmask
            if take.any():
                idx = np.where(take)[0]
                tokens[idx] = xg[idx]
                for i in idx:
                    sequences[i] = seqs[i]
                valid[idx] = True
            # for still-invalid slots, keep the freshest candidate (so tokens stay consistent)
            still = np.where(~valid)[0]
            if len(still):
                tokens[still] = xg[still]
                for i in still:
                    sequences[i] = seqs[i]
        if valid.all():
            break
    stats = {"best_of_n_draws_used": draws_used}
    return tokens, sequences, valid, stats


# --------------------------------------------------------------------------- #
#  Public dispatch                                                              #
# --------------------------------------------------------------------------- #
STRATEGY_PRESETS: Dict[str, Dict] = {
    "baseline":       dict(),
    "more_steps":     dict(steps_per_token=1.0),
    "top_p":          dict(top_p=0.9),
    "nucleus":        dict(top_p=0.9),
    "top_k":          dict(top_k=20),
    "low_temp":       dict(temperature=0.7),
    "remask":         dict(remask_rounds=3, remask_frac=0.25, remask_steps=32),
    "best_of_n":      dict(best_of_n=4),
    "nucleus_remask": dict(top_p=0.9, remask_rounds=3, remask_frac=0.25, remask_steps=32),
}


def available_strategies() -> List[str]:
    return list(STRATEGY_PRESETS.keys())


@torch.no_grad()
def generate(model, tokenizer, analyzer, batch_size: int, length: int,
             strategy: str = "baseline",
             num_steps: int = 128, eps: float = 1e-5,
             temperature: Optional[float] = None,
             top_p: Optional[float] = None,
             top_k: Optional[int] = None,
             steps_per_token: Optional[float] = None,
             remask_rounds: Optional[int] = None,
             remask_frac: Optional[float] = None,
             remask_steps: Optional[int] = None,
             best_of_n: Optional[int] = None,
             verbose: bool = False,
             ) -> Tuple[torch.Tensor, List[str], np.ndarray, Dict]:
    """Generate ``batch_size`` peptides of ``length`` tokens with the chosen strategy.

    Returns
    -------
    tokens     : LongTensor (batch_size, length)  final token ids
    sequences  : list[str]                         decoded SMILES
    valid_mask : np.ndarray[bool] (batch_size,)    analyzer.is_peptide per sequence
    stats      : dict                              metrics (valid_rate, timing, knobs, ...)
    """
    if strategy not in STRATEGY_PRESETS:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {available_strategies()}")

    # preset provides defaults; any explicitly-passed (non-None) kwarg overrides it
    cfg = dict(temperature=1.0, top_p=None, top_k=None, steps_per_token=None,
               remask_rounds=0, remask_frac=0.25, remask_steps=32, best_of_n=1)
    cfg.update(STRATEGY_PRESETS[strategy])
    explicit = dict(temperature=temperature, top_p=top_p, top_k=top_k,
                    steps_per_token=steps_per_token, remask_rounds=remask_rounds,
                    remask_frac=remask_frac, remask_steps=remask_steps, best_of_n=best_of_n)
    for key, val in explicit.items():
        if val is not None:
            cfg[key] = val

    # steps scale with length for `more_steps` (>= base num_steps)
    eff_steps = int(num_steps)
    if cfg["steps_per_token"] is not None:
        eff_steps = max(eff_steps, int(round(cfg["steps_per_token"] * length)))

    transform = make_transform(cfg["temperature"], cfg["top_p"], cfg["top_k"])

    t0 = time.time()
    if int(cfg["best_of_n"]) > 1:
        tokens, sequences, valid, extra = _generate_best_of_n(
            model, tokenizer, analyzer, batch_size, length, eff_steps, eps, transform,
            int(cfg["remask_rounds"]), float(cfg["remask_frac"]), int(cfg["remask_steps"]),
            int(cfg["best_of_n"]))
        round_valid = [int(valid.sum())]
    else:
        tokens, sequences, valid, _conf, round_valid = _generate_once(
            model, tokenizer, analyzer, batch_size, length, eff_steps, eps, transform,
            int(cfg["remask_rounds"]), float(cfg["remask_frac"]), int(cfg["remask_steps"]))
        extra = {}
    wall = time.time() - t0

    valid_count = int(valid.sum())
    stats = {
        "strategy": strategy,
        "length": int(length),
        "batch_size": int(batch_size),
        "num_steps": int(eff_steps),
        "temperature": float(cfg["temperature"]),
        "top_p": cfg["top_p"],
        "top_k": cfg["top_k"],
        "remask_rounds": int(cfg["remask_rounds"]),
        "remask_frac": float(cfg["remask_frac"]),
        "remask_steps": int(cfg["remask_steps"]),
        "best_of_n": int(cfg["best_of_n"]),
        "valid_count": valid_count,
        "valid_rate": valid_count / max(batch_size, 1),
        "round_valid_counts": round_valid,   # valid count after each remask round (shows mechanism)
        "wall_time_s": round(wall, 2),
    }
    stats.update(extra)
    if verbose:
        print(f"[{strategy}] L={length} steps={eff_steps} "
              f"valid={valid_count}/{batch_size} ({stats['valid_rate']:.1%}) "
              f"rounds={round_valid} {wall:.1f}s")
    return tokens, sequences, valid, stats


# --------------------------------------------------------------------------- #
#  Model construction helpers                                                   #
# --------------------------------------------------------------------------- #
def build_random_model(device="cpu", hidden_size: int = 768, n_layers: int = 8,
                       n_heads: int = 8, tokenizer=None, base_path: Optional[str] = None):
    """Construct a RANDOM-init ``Diffusion`` (no checkpoint) for CPU development /
    relative benchmarking.  Same architecture/config path as ``inference.load_model``
    minus the weight load, so samplers exercised here transfer unchanged to the real
    checkpoint.  Reduce hidden_size/n_layers for fast CPU benchmarks (yields are
    garbage either way with random init -- only relative trends are meaningful).
    """
    import os
    from configs.finetune_config import (DiffusionConfig, RoFormerConfig, NoiseConfig,
                                          TrainingConfig, SamplingConfig, EvalConfig,
                                          OptimConfig, MCTSConfig)
    from models.diffusion import Diffusion
    if tokenizer is None:
        from training.finetune_utils import load_tokenizer
        # default to this module's own directory (repo root) so it works from any cwd
        if base_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
        tokenizer = load_tokenizer(base_path)
    dev = torch.device(device)
    cfg = DiffusionConfig(
        roformer=RoFormerConfig(hidden_size=hidden_size, n_layers=n_layers, n_heads=n_heads),
        noise=NoiseConfig(), training=TrainingConfig(sampling_eps=1e-3),
        sampling=SamplingConfig(steps=128, sampling_eps=1e-3), eval_cfg=EvalConfig(),
        optim=OptimConfig(lr=3e-4), mcts=MCTSConfig())
    model = Diffusion(config=cfg, tokenizer=tokenizer, device=dev).to(dev)
    model.eval()
    model.tokenizer = tokenizer
    return model, tokenizer
