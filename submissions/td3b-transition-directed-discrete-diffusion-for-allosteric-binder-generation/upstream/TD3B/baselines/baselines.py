import logging
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_EPS = 1e-5
logger = logging.getLogger(__name__)


def _sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
    gumbel = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel).argmax(dim=-1).to(dtype=torch.long)


def _normalize_probs(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def _safe_resample_weights(weights: torch.Tensor) -> torch.Tensor:
    if weights.numel() == 0:
        return weights
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    total = weights.sum()
    if not torch.isfinite(total) or total <= 0:
        return torch.full_like(weights, 1.0 / weights.numel())
    return weights / total


def _sequence_logprob(
    probs: torch.Tensor,
    x_next: torch.Tensor,
    x_current: torch.Tensor,
    mask_idx: int,
) -> torch.Tensor:
    gather = probs.gather(-1, x_next.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
    mask = (x_current == mask_idx).to(gather.dtype)
    return (gather.log() * mask).sum(dim=-1)


def _transition_probs_from_logits(
    log_probs: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    mask_idx: int,
) -> torch.Tensor:
    change_prob_t = t[:, None, None]
    change_prob_s = (t - dt)[:, None, None]
    q_xs = log_probs.exp() * (change_prob_t - change_prob_s)
    q_xs[:, :, mask_idx] = change_prob_s[:, :, 0]
    return q_xs


def _sample_from_q(
    q_probs: torch.Tensor,
    x_current: torch.Tensor,
    mask_idx: int,
) -> torch.Tensor:
    x_changed = _sample_categorical(q_probs)
    copy_flag = (x_current != mask_idx)
    return torch.where(copy_flag, x_current, x_changed)


def _protein_tokens_to_device(tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tokens.device != device:
        return tokens.to(device)
    return tokens


def _tokens_to_one_hot(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return F.one_hot(tokens, num_classes=vocab_size).float()


def _decode_sequences(tokenizer, token_ids: torch.Tensor) -> list:
    return tokenizer.batch_decode(token_ids)


def _affinity_from_scoring(
    scoring_fn: Callable,
    sequences: list,
    device: torch.device,
    protein_seq: Optional[str] = None,
) -> torch.Tensor:
    if protein_seq is not None:
        try:
            scores = scoring_fn(sequences, protein_seq)
        except TypeError:
            try:
                scores = scoring_fn(sequences, prot_seq=protein_seq)
            except TypeError:
                scores = scoring_fn(sequences)
    else:
        scores = scoring_fn(sequences)
    if isinstance(scores, tuple):
        scores = scores[0]
    scores = np.asarray(scores)
    if scores.ndim == 1:
        affinity = scores
    else:
        affinity = scores[:, 0]
    return torch.as_tensor(affinity, device=device, dtype=torch.float32)


def _roformer_hidden_from_inputs(
    base_model,
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    outputs = base_model.backbone.model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    return outputs.hidden_states[-1]


def _logits_from_inputs(
    base_model,
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    outputs = base_model.backbone.model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        output_hidden_states=False,
        return_dict=True,
    )
    return outputs.logits


@dataclass
class RewardInputs:
    protein_tokens: torch.Tensor
    d_star: float
    protein_seq: str


class RewardWrapper:
    def __init__(
        self,
        scoring_fn: Callable,
        direction_oracle: torch.nn.Module,
        base_model,
        tokenizer,
        reward_inputs: RewardInputs,
        device: torch.device,
        fast_direction: bool = False,
        reward_alpha: float = 0.1,
    ):
        self.scoring_fn = scoring_fn
        self.direction_oracle = direction_oracle
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.reward_inputs = reward_inputs
        self.device = device
        self.fast_direction = fast_direction
        self.reward_alpha = reward_alpha
        self._supports_hidden_direction = all(
            hasattr(direction_oracle, attr)
            for attr in ("protein_embedder", "fusion", "classifier")
        )
        self._supports_predict = hasattr(direction_oracle, "predict_with_confidence")
        if self.fast_direction and not self._supports_hidden_direction:
            logger.warning("fast_direction requested but oracle lacks hidden-direction modules; disabling fast_direction.")
            self.fast_direction = False
        self._protein_emb_cache = None
        if self.reward_inputs.protein_seq is None:
            raise ValueError("RewardInputs.protein_seq is required for conditioned sampling.")

    def _protein_emb(self, batch_size: int) -> torch.Tensor:
        if not self._supports_hidden_direction:
            raise RuntimeError("direction_oracle does not support hidden-direction inference.")
        if self._protein_emb_cache is None:
            prot_tokens = _protein_tokens_to_device(self.reward_inputs.protein_tokens, self.device)
            prot_emb = self.direction_oracle.protein_embedder(prot_tokens)
            self._protein_emb_cache = prot_emb
        return self._protein_emb_cache.expand(batch_size, -1)

    def _direction_from_hidden(
        self,
        hidden: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self._supports_hidden_direction:
            raise RuntimeError("direction_oracle does not support hidden-direction inference.")
        mask = attn_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        protein_emb = self._protein_emb(pooled.size(0))
        fused = self.direction_oracle.fusion(pooled, protein_emb)
        return self.direction_oracle.classifier(fused).squeeze(-1)

    def _direction_from_probs(
        self,
        y_probs: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self.direction_oracle, "predict_from_probs"):
            prot_tokens = _protein_tokens_to_device(self.reward_inputs.protein_tokens, self.device)
            return self.direction_oracle.predict_from_probs(y_probs, prot_tokens, attn_mask)
        if not self._supports_hidden_direction:
            token_ids = y_probs.argmax(dim=-1)
            return self._direction_from_tokens(token_ids)
        if self.fast_direction:
            emb_weight = self.base_model.backbone.model.roformer.embeddings.word_embeddings.weight
            inputs_embeds = y_probs @ emb_weight
            hidden = inputs_embeds
        else:
            emb_weight = self.base_model.backbone.model.roformer.embeddings.word_embeddings.weight
            inputs_embeds = y_probs @ emb_weight
            hidden = _roformer_hidden_from_inputs(
                self.base_model,
                inputs_embeds=inputs_embeds,
                attn_mask=attn_mask,
            )
        return self._direction_from_hidden(hidden, attn_mask)

    def _direction_from_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        prot_tokens = _protein_tokens_to_device(self.reward_inputs.protein_tokens, self.device)
        if prot_tokens.dim() == 2 and prot_tokens.size(0) == 1:
            prot_tokens = prot_tokens.expand(token_ids.size(0), -1)
        if self._supports_predict:
            direction, _ = self.direction_oracle.predict_with_confidence(token_ids, prot_tokens)
            return direction
        return self.direction_oracle(token_ids, prot_tokens)

    def _gated_reward(self, affinity: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        d_star = torch.as_tensor(self.reward_inputs.d_star, device=self.device, dtype=direction.dtype)
        directional_score = (direction - 0.5) * d_star
        gate = torch.sigmoid(directional_score / self.reward_alpha)
        return affinity * gate

    def evaluate_tokens(self, token_ids: torch.Tensor, attn_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        sequences = _decode_sequences(self.tokenizer, token_ids)
        affinity = _affinity_from_scoring(
            self.scoring_fn,
            sequences,
            self.device,
            protein_seq=self.reward_inputs.protein_seq,
        )
        with torch.no_grad():
            direction = self._direction_from_tokens(token_ids)
        gated_reward = self._gated_reward(affinity, direction)
        return {
            "sequences": sequences,
            "affinity": affinity,
            "direction": direction,
            "gated_reward": gated_reward,
        }

    def reward_from_tokens(
        self,
        token_ids: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        sequences = _decode_sequences(self.tokenizer, token_ids)
        affinity = _affinity_from_scoring(
            self.scoring_fn,
            sequences,
            self.device,
            protein_seq=self.reward_inputs.protein_seq,
        )
        with torch.no_grad():
            direction = self._direction_from_tokens(token_ids)
        return self._gated_reward(affinity, direction)

    def reward_from_probs(
        self,
        y_probs: torch.Tensor,
        token_ids_for_affinity: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        affinity = None
        if hasattr(self.scoring_fn, "forward_from_probs"):
            try:
                affinity = self.scoring_fn.forward_from_probs(
                    y_probs,
                    attn_mask,
                    prot_seq=self.reward_inputs.protein_seq,
                )
            except Exception as exc:
                logger.warning("Differentiable affinity failed; falling back to argmax. Error: %s", exc)
                affinity = None
        if affinity is None:
            sequences = _decode_sequences(self.tokenizer, token_ids_for_affinity)
            affinity = _affinity_from_scoring(
                self.scoring_fn,
                sequences,
                self.device,
                protein_seq=self.reward_inputs.protein_seq,
            )
        direction = self._direction_from_probs(y_probs, attn_mask)
        return self._gated_reward(affinity, direction)


class PepTuneSampler:
    def __init__(
        self,
        base_model,
        reward_fn: RewardWrapper,
        seq_length: int,
        num_steps: int,
        mcts_iterations: int,
        num_children: int,
        sample_prob_weight: float,
        invalid_penalty: float,
        pareto_max_size: Optional[int],
        eps: float,
    ):
        from mcts.peptide_mcts import Node, updateParetoFront
        from utils.app import PeptideAnalyzer

        self.base_model = base_model
        self.reward_fn = reward_fn
        self.seq_length = seq_length
        self.num_steps = num_steps
        self.mcts_iterations = mcts_iterations
        self.num_children = num_children
        self.sample_prob_weight = sample_prob_weight
        self.invalid_penalty = invalid_penalty
        self.pareto_max_size = pareto_max_size
        self.eps = eps

        self.device = base_model.device
        self.mask_idx = base_model.mask_index
        self.tokenizer = base_model.tokenizer
        self.analyzer = PeptideAnalyzer()
        self.Node = Node
        self.updateParetoFront = updateParetoFront

        self.timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        self.dt = torch.as_tensor((1 - eps) / num_steps, device=self.device)
        self.args = SimpleNamespace(
            num_obj=1,
            total_num_steps=num_steps,
            seq_length=seq_length,
            num_children=num_children,
        )

    def _init_root(self):
        masked_seq = torch.full((self.seq_length,), self.mask_idx, device=self.device, dtype=torch.long)
        attn_mask = torch.ones_like(masked_seq, device=self.device)
        tokens = {"seqs": masked_seq, "attention_mask": attn_mask}
        return self.Node(
            args=self.args,
            tokens=tokens,
            log_rnd=torch.zeros((), device=self.device),
            log_policy_step=torch.zeros((), device=self.device),
            log_pretrained_step=torch.zeros((), device=self.device),
            totalReward=np.zeros(self.args.num_obj),
            timestep=0,
        )

    def _select(self, root):
        node = root
        while True:
            node, status = node.selectNode()
            if status != 3:
                return node, status

    def _update_pareto(self, pareto_front, pareto_tokens, seq, token_ids, score_vector):
        pareto_front = self.updateParetoFront(
            pareto_front,
            seq,
            score_vector,
            totalSize=self.pareto_max_size,
        )
        pareto_tokens = {k: pareto_tokens[k] for k in pareto_front if k in pareto_tokens}
        if seq in pareto_front:
            pareto_tokens[seq] = token_ids.detach().clone()
        return pareto_front, pareto_tokens

    def _expand(self, parent, pareto_front, pareto_tokens):
        parent_tokens = parent.tokens["seqs"].to(self.device)
        attn_mask = parent.tokens["attention_mask"].to(self.device)
        t = self.timesteps[parent.timestep] * torch.ones(1, 1, device=self.device)

        with torch.no_grad():
            _, x_children, log_policy_step, log_pretrained_step = self.base_model.batch_mcts_reverse_step(
                token_array=parent_tokens,
                t=t,
                dt=self.dt,
                batch_size=self.num_children,
                pretrained=self.base_model,
            )

        child_log_rnd = parent.log_rnd + (log_pretrained_step - log_policy_step)
        log_policy_step = log_policy_step * self.sample_prob_weight

        x_rollout = x_children
        t_step = self.timesteps[parent.timestep] * torch.ones(self.num_children, 1, device=self.device)
        for i in range(1, self.num_steps - parent.timestep):
            t_step = self.timesteps[parent.timestep + i] * torch.ones(self.num_children, 1, device=self.device)
            with torch.no_grad():
                _, x_next, _, _ = self.base_model.mcts_reverse_step(
                    x_rollout,
                    t=t_step,
                    dt=self.dt,
                    pretrained=self.base_model,
                )
            x_rollout = x_next

        if (x_rollout == self.mask_idx).any().item():
            with torch.no_grad():
                _, x_next, _, _ = self.base_model.mcts_noise_removal(
                    x_rollout,
                    t=t_step,
                    dt=self.dt,
                    pretrained=self.base_model,
                )
            x_rollout = x_next

        sequences = self.tokenizer.batch_decode(x_rollout)
        valid_mask = [self.analyzer.is_peptide(seq) for seq in sequences]

        reward_values = np.full(self.num_children, -float(self.invalid_penalty), dtype=np.float32)
        if any(valid_mask):
            valid_tokens = x_rollout[valid_mask]
            valid_sequences = [seq for seq, keep in zip(sequences, valid_mask) if keep]
            affinity = _affinity_from_scoring(
                self.reward_fn.scoring_fn,
                valid_sequences,
                self.device,
                protein_seq=self.reward_fn.reward_inputs.protein_seq,
            )
            with torch.no_grad():
                direction = self.reward_fn._direction_from_tokens(valid_tokens)
            gated_reward = self.reward_fn._gated_reward(affinity, direction)
            d_star = self.reward_fn.reward_inputs.d_star
            dir_score = (direction - 0.5) * d_star

            for idx, seq in enumerate(valid_sequences):
                score_vector = np.array(
                    [float(affinity[idx].item()), float(dir_score[idx].item())],
                    dtype=np.float32,
                )
                pareto_front, pareto_tokens = self._update_pareto(
                    pareto_front,
                    pareto_tokens,
                    seq,
                    valid_tokens[idx],
                    score_vector,
                )

            reward_values[np.array(valid_mask)] = gated_reward.detach().cpu().numpy()

        reward_vectors = []
        for i in range(self.num_children):
            child_tokens = {"seqs": x_children[i].to(dtype=torch.long), "attention_mask": attn_mask}
            reward_vec = np.array([float(reward_values[i])], dtype=np.float32)
            parent.addChildNode(
                tokens=child_tokens,
                log_rnd=child_log_rnd[i],
                log_policy_step=log_policy_step[i],
                log_pretrained_step=log_pretrained_step[i],
                totalReward=reward_vec,
            )
            reward_vectors.append(reward_vec)

        avg_reward = np.mean(np.stack(reward_vectors, axis=0), axis=0)
        node = parent
        while node:
            node.updateNode(avg_reward)
            node = node.parentNode

        return pareto_front, pareto_tokens

    def _select_from_pareto(self, pareto_front, pareto_tokens, batch_size):
        if not pareto_front:
            return self.base_model.sample_prior(batch_size, self.seq_length).to(self.device)

        seqs = list(pareto_front.keys())
        scores = np.stack([pareto_front[seq] for seq in seqs], axis=0)
        affinity = scores[:, 0]
        dir_score = scores[:, 1]
        gate = 1.0 / (1.0 + np.exp(-dir_score / max(self.reward_fn.reward_alpha, 1e-6)))
        gated = affinity * gate
        order = np.argsort(-gated)

        if len(order) >= batch_size:
            selected = [seqs[i] for i in order[:batch_size]]
        else:
            repeats = np.random.choice(order, size=batch_size, replace=True)
            selected = [seqs[i] for i in repeats]

        tokens = [pareto_tokens[seq] for seq in selected]
        return torch.stack(tokens, dim=0).to(self.device)

    def sample(self, batch_size):
        self.base_model.eval()
        root = self._init_root()
        pareto_front = {}
        pareto_tokens = {}

        for _ in range(self.mcts_iterations):
            leaf, status = self._select(root)
            if status == 1:
                continue
            pareto_front, pareto_tokens = self._expand(leaf, pareto_front, pareto_tokens)

        return self._select_from_pareto(pareto_front, pareto_tokens, batch_size)


def _logits_and_probs_from_tokens(
    base_model,
    token_ids: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    logits = _logits_from_inputs(base_model, input_ids=token_ids, attn_mask=attn_mask)
    log_probs = base_model.subs_parameterization(logits, token_ids)
    return log_probs


def _logits_and_probs_from_one_hot(
    base_model,
    y_one_hot: torch.Tensor,
    token_ids: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    emb_weight = base_model.backbone.model.roformer.embeddings.word_embeddings.weight
    inputs_embeds = y_one_hot @ emb_weight
    logits = _logits_from_inputs(base_model, inputs_embeds=inputs_embeds, attn_mask=attn_mask)
    log_probs = base_model.subs_parameterization(logits, token_ids)
    return log_probs


def classifier_guidance(
    base_model,
    reward_fn: RewardWrapper,
    batch_size: int,
    seq_length: int,
    num_steps: int,
    guidance_scale: float,
    eps: float = DEFAULT_EPS,
    guidance_steps: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    device = base_model.device
    mask_idx = base_model.mask_index
    vocab_size = base_model.vocab_size
    x = base_model.sample_prior(batch_size, seq_length).to(device)
    attn_mask = torch.ones_like(x, device=device)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    dt = torch.as_tensor((1 - eps) / num_steps, device=device)

    guidance_enabled = True
    for step in range(num_steps):
        t = timesteps[step].repeat(batch_size)
        use_guidance = guidance_enabled and (guidance_steps is None or step >= num_steps - guidance_steps)
        if not use_guidance:
            log_probs = _logits_and_probs_from_tokens(base_model, x, attn_mask)
            q_base = _transition_probs_from_logits(log_probs, t, dt, mask_idx)
            x = _sample_from_q(q_base, x, mask_idx)
            continue

        y_one_hot = _tokens_to_one_hot(x, vocab_size).to(device)
        y_one_hot.requires_grad_(True)
        token_ids = x.detach()
        log_probs = _logits_and_probs_from_one_hot(base_model, y_one_hot, token_ids, attn_mask)
        y_probs = log_probs.exp()
        token_ids_for_affinity = y_probs.argmax(dim=-1).detach()
        reward = reward_fn.reward_from_probs(y_probs, token_ids_for_affinity, attn_mask)
        if not reward.requires_grad:
            if guidance_enabled:
                logger.warning(
                    "Reward does not require grad; disabling gradient guidance for classifier_guidance."
                )
            guidance_enabled = False
            q_base = _transition_probs_from_logits(log_probs, t, dt, mask_idx)
            x = _sample_from_q(q_base, x, mask_idx)
            continue
        reward.sum().backward()
        grad = y_one_hot.grad
        q_base = _transition_probs_from_logits(log_probs, t, dt, mask_idx)
        guidance = guidance_scale * (grad - grad[:, :, mask_idx].unsqueeze(-1))
        guidance = guidance.clamp(min=-50.0, max=50.0)
        q_guided = q_base * torch.exp(guidance)
        q_guided = _normalize_probs(q_guided)
        x = _sample_from_q(q_guided, x, mask_idx)

    return {"tokens": x}


def unguided_sampling(
    base_model,
    batch_size: int,
    seq_length: int,
    num_steps: int,
    eps: float = DEFAULT_EPS,
) -> Dict[str, torch.Tensor]:
    device = base_model.device
    mask_idx = base_model.mask_index
    x = base_model.sample_prior(batch_size, seq_length).to(device)
    attn_mask = torch.ones_like(x, device=device)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    dt = torch.as_tensor((1 - eps) / num_steps, device=device)

    for step in range(num_steps):
        t = timesteps[step].repeat(batch_size)
        log_probs = _logits_and_probs_from_tokens(base_model, x, attn_mask)
        q_base = _transition_probs_from_logits(log_probs, t, dt, mask_idx)
        x = _sample_from_q(q_base, x, mask_idx)

    return {"tokens": x}


def sequential_monte_carlo(
    base_model,
    reward_fn: RewardWrapper,
    batch_size: int,
    seq_length: int,
    num_steps: int,
    alpha: float,
    eps: float = DEFAULT_EPS,
) -> Dict[str, torch.Tensor]:
    device = base_model.device
    mask_idx = base_model.mask_index
    x = base_model.sample_prior(batch_size, seq_length).to(device)
    attn_mask = torch.ones_like(x, device=device)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    dt = torch.as_tensor((1 - eps) / num_steps, device=device)

    with torch.no_grad():
        r_current = reward_fn.reward_from_tokens(x, attn_mask).detach()
    for step in range(num_steps):
        t = timesteps[step].repeat(batch_size)
        log_probs = _logits_and_probs_from_tokens(base_model, x, attn_mask)
        q_base = _transition_probs_from_logits(log_probs, t, dt, mask_idx)
        x_next = _sample_from_q(q_base, x, mask_idx)

        with torch.no_grad():
            r_next = reward_fn.reward_from_tokens(x_next, attn_mask).detach()
        weights = torch.exp((r_next - r_current) / alpha).clamp_max(1e6)
        weights = _safe_resample_weights(weights)
        indices = torch.multinomial(weights, num_samples=batch_size, replacement=True)
        x = x_next[indices]
        r_current = r_next[indices]

    return {"tokens": x}


def twisted_diffusion_sampler(
    base_model,
    reward_fn: RewardWrapper,
    batch_size: int,
    seq_length: int,
    num_steps: int,
    guidance_scale: float,
    alpha: float,
    eps: float = DEFAULT_EPS,
    guidance_steps: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    device = base_model.device
    mask_idx = base_model.mask_index
    vocab_size = base_model.vocab_size
    x = base_model.sample_prior(batch_size, seq_length).to(device)
    attn_mask = torch.ones_like(x, device=device)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    dt = torch.as_tensor((1 - eps) / num_steps, device=device)

    with torch.no_grad():
        r_current = reward_fn.reward_from_tokens(x, attn_mask).detach()
    guidance_enabled = True
    for step in range(num_steps):
        t = timesteps[step].repeat(batch_size)
        use_guidance = guidance_enabled and (guidance_steps is None or step >= num_steps - guidance_steps)

        if use_guidance:
            y_one_hot = _tokens_to_one_hot(x, vocab_size).to(device)
            y_one_hot.requires_grad_(True)
            token_ids = x.detach()
            log_probs = _logits_and_probs_from_one_hot(base_model, y_one_hot, token_ids, attn_mask)
            y_probs = log_probs.exp()
            token_ids_for_affinity = y_probs.argmax(dim=-1).detach()
            reward = reward_fn.reward_from_probs(y_probs, token_ids_for_affinity, attn_mask)
            q_base = _transition_probs_from_logits(log_probs, t, dt, mask_idx)
            if not reward.requires_grad:
                if guidance_enabled:
                    logger.warning(
                        "Reward does not require grad; disabling gradient guidance for twisted_diffusion_sampler."
                    )
                guidance_enabled = False
                q_guided = q_base
            else:
                reward.sum().backward()
                grad = y_one_hot.grad
                guidance = guidance_scale * (grad - grad[:, :, mask_idx].unsqueeze(-1))
                guidance = guidance.clamp(min=-50.0, max=50.0)
                q_guided = q_base * torch.exp(guidance)
                q_guided = _normalize_probs(q_guided)
        else:
            log_probs = _logits_and_probs_from_tokens(base_model, x, attn_mask)
            q_base = _transition_probs_from_logits(log_probs, t, dt, mask_idx)
            q_guided = q_base

        x_next = _sample_from_q(q_guided, x, mask_idx)
        with torch.no_grad():
            r_next = reward_fn.reward_from_tokens(x_next, attn_mask).detach()

        logp_guided = _sequence_logprob(q_guided, x_next, x, mask_idx)
        logp_base = _sequence_logprob(q_base, x_next, x, mask_idx)
        weights = torch.exp((r_next - r_current) / alpha + (logp_base - logp_guided)).clamp_max(1e6)
        weights = _safe_resample_weights(weights)
        indices = torch.multinomial(weights, num_samples=batch_size, replacement=True)
        x = x_next[indices]
        r_current = r_next[indices]

    return {"tokens": x}


def peptune_mctg_sampling(
    base_model,
    reward_fn: RewardWrapper,
    batch_size: int,
    seq_length: int,
    num_steps: int,
    mcts_iterations: int,
    num_children: int,
    alpha: float,
    sample_prob_weight: float,
    invalid_penalty: float = 1.0,
    pareto_max_size: Optional[int] = None,
    eps: float = DEFAULT_EPS,
) -> Dict[str, torch.Tensor]:
    sampler = PepTuneSampler(
        base_model=base_model,
        reward_fn=reward_fn,
        seq_length=seq_length,
        num_steps=num_steps,
        mcts_iterations=mcts_iterations,
        num_children=num_children,
        sample_prob_weight=sample_prob_weight,
        invalid_penalty=invalid_penalty,
        pareto_max_size=pareto_max_size,
        eps=eps,
    )
    tokens = sampler.sample(batch_size=batch_size)
    return {"tokens": tokens}
