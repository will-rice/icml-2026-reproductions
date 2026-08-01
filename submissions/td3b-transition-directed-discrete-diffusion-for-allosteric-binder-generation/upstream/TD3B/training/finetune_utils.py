"""Utility functions for TD3B finetuning and sampling."""

import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.diffusion import Diffusion
from td3b.td3b_mcts import create_td3b_mcts
from td3b.td3b_scoring import TD3BRewardFunction
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from utils.utils import sample_categorical_logits

logger = logging.getLogger(__name__)

# Standard checkpoint keys to try when loading
CHECKPOINT_KEYS = ("state_dict", "model_state_dict")


def to_one_hot(x_idx, num_classes=4):
    oh = F.one_hot(x_idx.long(), num_classes=num_classes)
    return oh.float()


def rnd(model, reward_model, batch_size, scale=1, device="cuda:0"):
    r"""
    Run random order sampling and compute the RND $\log\frac{dP^*}{dP^u}$ along the trajectory
    reward_model: r(X)

    return:
    - x: the final samples, [B, D]
    - log_rnd: the log RND along this trajectory, [B]
    """
    if hasattr(model, "module"):
        model = model.module

    x = torch.full((batch_size, model.length), model.vocab_size - 1).to(device=device, dtype=torch.int64)
    batch_arange = torch.arange(batch_size, device=device)
    jump_pos = torch.rand(x.shape, device=device).argsort(dim=-1)
    # jump_times, jump_pos = torch.rand(x.shape, device=device).sort(dim=-1)
    # jump_times: Unif[0,1] in increasing order
    # jump_pos: random permutation of range(D)
    log_rnd = torch.zeros(batch_size, device=device)  # [B]
    for d in range(model.length - 1, -1, -1):
        # jump at time jump_times[:, d] at position jump_pos[:, d]
        logits = model(x)[:, :, :-1]  # [B, D, N-1]
        update = sample_categorical_logits(logits[batch_arange, jump_pos[:, d]])  # [B]
        if torch.is_grad_enabled():  # avoid issues with in-place operations
            x = x.clone()
        x[batch_arange, jump_pos[:, d]] = update
        log_rnd += -np.log(model.vocab_size - 1) - logits[batch_arange, jump_pos[:, d], update]
    log_rnd += scale * reward_model(x)  # [B]
    return x, log_rnd


@torch.no_grad()
def sampling(model, batch_size, rounds=1, device="cuda:0"):
    """Any order autoregressive sampling"""
    if hasattr(model, "module"):
        model = model.module
    batch_arange = torch.arange(batch_size, device=device)
    all_samples = []
    for _ in tqdm(range(rounds), leave=False):
        x = torch.full((batch_size, model.length), model.vocab_size - 1).to(device=device, dtype=torch.int64)
        jump_pos = torch.rand(x.shape, device=device).argsort(dim=-1)
        # jump_times, jump_pos = torch.rand(x.shape, device=device).sort(dim=-1)
        # jump_times: Unif[0,1] in increasing order
        # jump_pos: random permutation of range(D)
        for d in tqdm(range(model.length - 1, -1, -1), leave=False):
            # jump at time jump_times[:, d] at position jump_pos[:, d]
            logits = model.logits(x)[:, :, :-1]  # [B, D, N-1], not log-softmaxed but fine
            update = sample_categorical_logits(logits[batch_arange, jump_pos[:, d]])  # [B]
            x[batch_arange, jump_pos[:, d]] = update
        all_samples.append(x)
    return torch.cat(all_samples)  # (rounds * B, L)


def loss_ce(log_rnd):
    """Cross entropy loss KL(P^*||P^u)"""
    weights = log_rnd.detach().softmax(dim=-1)
    return (log_rnd * weights).sum()


def loss_lv(log_rnd):
    r"""Log variance loss Var_{P^\bar{u}}\log\frac{dP^*}{dP^u}"""
    return log_rnd.var()


def loss_re_rf(log_rnd, const=0):
    r"""Relative entropy loss KL(P^u||P^*) with REINFORCE trick"""
    return (-log_rnd * (-log_rnd.detach() + const)).mean()


def loss_wdce(
    policy_model,
    log_rnd,
    x,
    num_replicates=16,
    weight_func=lambda l: 1 / l,
    eps=1e-3,
    centering=False,
    attn_mask=None,
):
    r"""
    Weighted denoising cross entropy loss
    X_T ~ P^u_T and weights \log\frac{dP^*}{dP^u}(X)

    log_rnd: [B]; x: [B, L] (no mask)
    num_replicates: R, number of replicates of each row in x
    weight_func: w(lambda) for each sample, 1/lambda by default
    attn_mask: [B, L] attention mask (1 for real tokens, 0 for padding) - IMPORTANT for variable-length sequences
    """
    mask_index = policy_model.mask_index
    if hasattr(policy_model, "module"):
        policy_model = policy_model.module

    batch = x.repeat_interleave(num_replicates, dim=0)  # [B*R, L]

    batch_weights = log_rnd.detach_().softmax(dim=-1)  # [B*R]
    if centering:
        batch_weights = batch_weights - batch_weights.mean(dim=-1, keepdim=True)

    batch_weights = batch_weights.repeat_interleave(num_replicates, dim=0)

    lamda = torch.rand(batch.shape[0], device=batch.device)  # [B*R]
    lamda_weights = weight_func(lamda).clamp(max=1e5)  # [B*R]

    masked_index = torch.rand(*batch.shape, device=batch.device) < lamda[..., None]  # [B*R, D]
    perturbed_batch = torch.where(masked_index, mask_index, batch)

    # add time conditioning
    t = lamda
    sigma_t = -torch.log1p(-(1 - eps) * t)

    # Use provided attention mask or create default (all ones for fixed-length)
    if attn_mask is not None:
        attn_mask = attn_mask.repeat_interleave(num_replicates, dim=0).to(policy_model.device)
    else:
        attn_mask = torch.ones_like(perturbed_batch).to(policy_model.device)

    # compute logits
    logits = policy_model(perturbed_batch, attn_mask=attn_mask, sigma=sigma_t)
    losses = torch.zeros(*batch.shape, device=batch.device, dtype=logits.dtype)  # [B*R, D]
    losses[masked_index] = torch.gather(
        input=logits[masked_index], dim=-1, index=batch[masked_index][..., None]
    ).squeeze(-1)

    # Apply attention mask to exclude padding tokens from loss computation.
    losses = losses * attn_mask

    return -((losses.sum(dim=-1) * lamda_weights * batch_weights).mean())


def loss_dce(model, x, weight_func=lambda l: 1 / l):
    r"""
    Denoising cross entropy loss, x [B, D] are ground truth samples
    weight_func: w(lambda) for each sample, 1/lambda by default
    """
    lamda = torch.rand(x.shape[0], device=x.device)  # [B]
    lamda_weights = weight_func(lamda).clamp(max=1e5)  # [B]
    masked_index = torch.rand(*x.shape, device=x.device) < lamda[..., None]  # [B, D]
    perturbed_batch = torch.where(masked_index, model.vocab_size - 1, x)
    logits = model(perturbed_batch)
    losses = torch.zeros(*x.shape, device=x.device, dtype=logits.dtype)  # [B, D]
    losses[masked_index] = torch.gather(
        input=logits[masked_index], dim=-1, index=x[masked_index][..., None]
    ).squeeze(-1)
    return -((losses.sum(dim=-1) * lamda_weights).mean())


def load_tokenizer(base_path: str) -> SMILES_SPE_Tokenizer:
    """
    Load the peptide tokenizer from the standard location.

    Args:
        base_path: Base directory path (e.g., 'To Be Added')

    Returns:
        Loaded SMILES_SPE_Tokenizer instance

    Example:
        >>> tokenizer = load_tokenizer('To Be Added')
    """
    base_path = Path(base_path)
    vocab_path = base_path / "tokenizer" / "new_vocab.txt"
    spe_path = base_path / "tokenizer" / "new_splits.txt"

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    if not spe_path.exists():
        raise FileNotFoundError(f"SPE splits file not found: {spe_path}")

    tokenizer = SMILES_SPE_Tokenizer(str(vocab_path), str(spe_path))
    logger.info("Loaded tokenizer with vocab_size=%s", tokenizer.vocab_size)

    return tokenizer


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: torch.device,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model weights from checkpoint with automatic key detection.

    Handles different checkpoint formats:
    - {'state_dict': ...}
    - {'model_state_dict': ...}
    - Direct state_dict

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        device: Device to load checkpoint onto
        strict: Whether to strictly enforce state_dict keys match

    Returns:
        Full checkpoint dictionary (for accessing metadata like epoch, config, etc.)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails

    Example:
        >>> checkpoint = load_checkpoint('model.ckpt', model, device, strict=False)
        >>> if 'epoch' in checkpoint:
        >>>     print(f"Loaded from epoch {checkpoint['epoch']}")
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint from: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try to find state_dict in standard checkpoint keys
    state_dict = None
    for key in CHECKPOINT_KEYS:
        if key in checkpoint:
            state_dict = checkpoint[key]
            logger.info("Found state_dict at checkpoint key: '%s'", key)
            break

    # If not found in standard keys, assume checkpoint IS the state_dict
    if state_dict is None:
        state_dict = checkpoint
        logger.info("Loading checkpoint as direct state_dict")

    # Load state dict into model
    try:
        incompatible_keys = model.load_state_dict(state_dict, strict=strict)
        if not strict and (incompatible_keys.missing_keys or incompatible_keys.unexpected_keys):
            logger.warning("Incompatible keys when loading checkpoint:")
            if incompatible_keys.missing_keys:
                logger.warning("  Missing keys: %s...", incompatible_keys.missing_keys[:5])
            if incompatible_keys.unexpected_keys:
                logger.warning("  Unexpected keys: %s...", incompatible_keys.unexpected_keys[:5])
        else:
            logger.info("Checkpoint loaded successfully")
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint: {exc}")

    return checkpoint


def initialize_device(device_str: str = "cuda") -> torch.device:
    """
    Initialize compute device with fallback to CPU if CUDA unavailable or invalid.

    Args:
        device_str: Requested device ('cuda', 'cuda:0', 'cpu', or 'auto')

    Returns:
        Torch device object

    Example:
        >>> device = initialize_device('cuda')
        >>> print(device)  # cuda:0 or cpu
    """
    if device_str is None or str(device_str).lower() == "auto":
        device_str = "cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"

    try:
        device = torch.device(device_str)
    except Exception as exc:
        logger.warning("Invalid device '%s': %s. Falling back to CPU.", device_str, exc)
        return torch.device("cpu")

    if device.type != "cuda":
        logger.info("Using device: %s", device)
        return device

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        logger.warning("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")

    index = device.index if device.index is not None else 0
    if index < 0 or index >= torch.cuda.device_count():
        logger.warning(
            "CUDA device %s requested but only %d visible; using cuda:0",
            index,
            torch.cuda.device_count(),
        )
        device = torch.device("cuda:0")

    logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(device.index or 0))
    return device


def create_output_directory(base_path: str, run_name: str, add_timestamp: bool = True) -> str:
    """
    Create output directory for saving results.

    Args:
        base_path: Base directory (e.g., 'To Be Added')
        run_name: Name for this training run
        add_timestamp: Whether to append timestamp to run_name

    Returns:
        Path to created output directory

    Example:
        >>> save_path = create_output_directory('To Be Added', 'my_run')
        >>> # Creates: To Be Added
    """
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{run_name}_{timestamp}"
    else:
        dir_name = run_name

    output_dir = os.path.join(base_path, "results", dir_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Created output directory: %s", output_dir)
    return output_dir


def save_model(
    model: torch.nn.Module,
    save_path: str,
    config: Optional[Dict[str, Any]] = None,
    epoch: Optional[int] = None,
    optimizer_state: Optional[Dict] = None,
) -> None:
    """
    Save model checkpoint with optional metadata.

    Args:
        model: Model to save
        save_path: Path to save checkpoint
        config: Optional configuration dictionary to save
        epoch: Optional epoch number
        optimizer_state: Optional optimizer state dict

    Example:
        >>> save_model(model, 'checkpoint.ckpt', config=vars(args), epoch=10)
    """
    checkpoint = {"model_state_dict": model.state_dict()}

    if config is not None:
        checkpoint["config"] = config
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state

    torch.save(checkpoint, save_path)
    logger.info("Model saved: %s", save_path)


def setup_wandb(project: str, name: str, config: Dict[str, Any], entity: Optional[str] = None) -> None:
    """
    Initialize Weights & Biases logging.

    Args:
        project: W&B project name
        name: Run name
        config: Configuration dictionary to log
        entity: Optional W&B team/entity name

    Example:
        >>> setup_wandb('my-project', 'run1', vars(args), entity='my-team')
    """
    wandb_config = {
        "project": project,
        "name": name,
        "config": config,
    }

    if entity:
        wandb_config["entity"] = entity

    wandb.init(**wandb_config)
    logger.info("Initialized W&B: project=%s, run=%s", project, name)


def cleanup_wandb() -> None:
    """Finish W&B logging session."""
    wandb.finish()
    logger.info("Finished W&B logging")


def get_mask_index(tokenizer: SMILES_SPE_Tokenizer) -> int:
    """
    Get mask token index from tokenizer.

    Args:
        tokenizer: Peptide tokenizer

    Returns:
        Mask token ID

    Note:
        Standardizes mask index retrieval across different code paths.
    """
    if hasattr(tokenizer, "mask_token_id"):
        return tokenizer.mask_token_id
    return tokenizer.convert_tokens_to_ids(tokenizer.mask_token)


def create_mcts_instance(
    args,
    policy_model: Diffusion,
    reward_function: TD3BRewardFunction,
    tokenizer: SMILES_SPE_Tokenizer,
    buffer_size: Optional[int] = None,
) -> Any:
    """
    Create TD3B MCTS instance with standardized configuration.

    Args:
        args: Training arguments
        policy_model: Diffusion policy model
        reward_function: TD3B reward function
        tokenizer: Peptide tokenizer
        buffer_size: Optional buffer size (uses args.buffer_size if None)

    Returns:
        TD3B_MCTS instance

    Example:
        >>> mcts = create_mcts_instance(args, model, reward_func, tokenizer)
    """
    if hasattr(args, "no_mcts") and args.no_mcts:
        logger.info("MCTS disabled (--no_mcts flag)")
        return None

    # Get mask index using standardized method
    mask_index = get_mask_index(tokenizer)

    # Use provided buffer_size or fall back to args
    if buffer_size is None:
        buffer_size = getattr(args, "buffer_size", 50)

    mcts = create_td3b_mcts(
        args=args,
        diffusion_model=policy_model,
        td3b_reward_function=reward_function,
        alpha=getattr(args, "alpha", 0.1),
        mask_index=mask_index,
        buffer_size=buffer_size,
        tokenizer=tokenizer,
    )

    logger.info("Created TD3B MCTS (buffer_size=%s, alpha=%s)", buffer_size, args.alpha)
    return mcts


def create_reward_function(
    affinity_predictor,
    directional_oracle,
    target_direction: float,
    target_protein_tokens: torch.Tensor,
    tokenizer: SMILES_SPE_Tokenizer,
    device: torch.device,
    min_affinity_threshold: float = 0.0,
    use_confidence_weighting: bool = True,
    temperature: float = 0.1,
) -> TD3BRewardFunction:
    """
    Create TD3B reward function with standardized parameters.

    Args:
        affinity_predictor: Binding affinity prediction model
        directional_oracle: Directional prediction oracle
        target_direction: Target direction (1.0 for agonist, -1.0 for antagonist)
        target_protein_tokens: Protein target tokens
        tokenizer: Peptide tokenizer
        device: Compute device
        min_affinity_threshold: Minimum affinity for allosteric control
        use_confidence_weighting: Whether to use confidence weighting
        temperature: Temperature for sigmoid gating

    Returns:
        TD3BRewardFunction instance

    Example:
        >>> reward_func = create_reward_function(
        ...     affinity_pred, oracle, 1.0, target_tokens,
        ...     tokenizer, device, min_affinity_threshold=0.5
        ... )
    """
    reward_func = TD3BRewardFunction(
        affinity_predictor=affinity_predictor,
        directional_oracle=directional_oracle,
        target_direction=target_direction,
        target_protein_tokens=target_protein_tokens,
        peptide_tokenizer=tokenizer,
        device=device,
        min_affinity_threshold=min_affinity_threshold,
        use_confidence_weighting=use_confidence_weighting,
        temperature=temperature,
    )

    logger.info(
        "Created TD3B reward function (d*=%s, threshold=%s)", target_direction, min_affinity_threshold
    )
    return reward_func


def log_gpu_memory(stage: str = "") -> None:
    """
    Log current GPU memory usage.

    Args:
        stage: Optional stage description for logging context

    Example:
        >>> log_gpu_memory("After model loading")
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        stage_str = f" [{stage}]" if stage else ""
        logger.info(
            "GPU Memory%s: %.2fGB allocated, %.2fGB reserved",
            stage_str,
            allocated,
            reserved,
        )


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)

    Example:
        >>> total, trainable = count_parameters(model)
        >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
