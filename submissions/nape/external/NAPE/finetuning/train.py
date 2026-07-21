"""
Causal-LM Finetuning for Spreadsheet Operation Completion
==========================================================

Trains a small causal language model on spreadsheet operation sequences
using standard next-token prediction.  Training data is formatted to be
**byte-identical** to what the ``CompletionSolver`` sends to
``LocalModelAdapter.complete()`` at inference time.

Supports:
    - Full finetune  (default for ≤ 1 B models)
    - Optional LoRA  via ``peft``
    - Mixed precision (fp16 / bf16)
    - Gradient checkpointing
    - TensorBoard + optional Weights & Biases logging
    - Best-checkpoint selection by validation loss
    - Configurable prompt template
    - On-the-fly preprocessing (no re-export when changing params)
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FINETUNING_ROOT = Path(__file__).resolve().parent

# Ensure package imports work
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_FINETUNING_ROOT) not in sys.path:
    sys.path.insert(0, str(_FINETUNING_ROOT))

# Default prompt template — must contain {actions} placeholder
DEFAULT_PROMPT_TEMPLATE = (
    "Complete the sequence of actions to build the following "
    "spreadsheet by identifying and extending key patterns.\n\n{actions}"
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = dict(
    # Model
    model="Qwen/Qwen2.5-0.5B",
    # Data
    data_dir="results/generation/train_data",
    processed_data_path="finetuning/processed_data/sequences.jsonl",
    max_percentile=95.0,
    eval_split=0.05,
    seed=42,
    # Pre-processing  (must match evaluation solver settings)
    preprocessing=dict(
        remove_sheet_name=True,
        context_shortening=dict(
            enabled=True,
            max_chars=32,
            corner_cells_dim=2,
        ),
    ),
    # Prompt template — must contain {actions} placeholder
    prompt_template=DEFAULT_PROMPT_TEMPLATE,
    # Example construction
    max_context_ops=128,
    example_stride=64,
    max_seq_len=2048,
    include_sheet_name=False,
    # Training
    epochs=3,
    batch_size=4,
    gradient_accumulation_steps=8,
    lr=5e-5,
    warmup_ratio=0.05,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    # LoRA (optional)
    use_lora=False,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=None,
    # Prediction accuracy eval
    eval_test_cases=None,  # path to JSON test cases (relative to project root)
    eval_before_training=True,
    # Output & logging
    output_dir="finetuning/results",
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=3,
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="tensorboard",
    wandb_project=None,
    wandb_run_name=None,
    # ---- Speed optimizations -----------------------------------------------
    # Group examples by length to reduce padding waste (~1.26x speedup).
    # Set to True for production runs; False for debug/reproducibility.
    group_by_length=False,
    # Apply Liger Kernel fused ops (Llama-arch only — works for SmolLM2/Qwen2).
    # Saves ~3.6 GB of memory on 360M model and gives a small speed bump.
    # When combined with group_by_length, total speedup is ~1.68x.
    use_liger_kernel=False,
)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        elif v is not None:
            merged[k] = v
    return merged


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config merged with defaults."""
    cfg = dict(_DEFAULTS)
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)
    return cfg


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override config values with explicitly-passed CLI flags."""
    overridable = [
        "model", "data_dir", "processed_data_path", "output_dir",
        "epochs", "batch_size", "lr", "max_seq_len", "max_context_ops",
        "example_stride", "seed", "fp16", "bf16",
        "gradient_accumulation_steps", "wandb_project", "report_to",
        "max_percentile",
    ]
    for key in overridable:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    if getattr(args, "use_lora", None):
        cfg["use_lora"] = True
    if getattr(args, "lora_rank", None) is not None:
        cfg["lora_rank"] = args.lora_rank
    return cfg


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: Dict[str, Any]) -> None:
    """Run the full training pipeline."""
    # ---- Apply Liger Kernel BEFORE importing model classes -----------------
    # Liger monkey-patches Llama-arch model code. Must be applied before
    # AutoModelForCausalLM resolves the model class.
    if cfg.get("use_liger_kernel"):
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_llama
            apply_liger_kernel_to_llama(
                rope=True,
                cross_entropy=False,
                fused_linear_cross_entropy=True,
                rms_norm=True,
                swiglu=True,
            )
            logger.info("Liger Kernel applied (Llama-arch fused ops + FLCE).")
        except ImportError as e:
            logger.warning(
                "use_liger_kernel=True but liger-kernel is not installed: %s. "
                "Falling back to vanilla model. Run `pip install liger-kernel` "
                "to enable.", e,
            )

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from data_preparation import (
        filter_by_percentile,
        load_jsonl,
        load_raw_sequences,
        save_jsonl,
    )
    from dataset import CausalLMCollator, OperationSequenceDataset

    # ---- Resolve paths relative to project root -----------------------------
    data_dir = _PROJECT_ROOT / cfg["data_dir"]
    processed_path = _PROJECT_ROOT / cfg["processed_data_path"]
    output_dir = _PROJECT_ROOT / cfg["output_dir"]

    # ---- Pre-processing config ----------------------------------------------
    prep = cfg.get("preprocessing", {})
    shortening = prep.get("context_shortening", {})

    # ---- Step 1: Ensure JSONL data exists -----------------------------------
    if not processed_path.exists():
        logger.info(
            "JSONL not found at %s — exporting raw sequences…", processed_path
        )
        data = load_raw_sequences(data_dir)
        data = filter_by_percentile(data, cfg.get("max_percentile", 95.0))
        save_jsonl(data, processed_path)
    else:
        data = load_jsonl(processed_path)

    # ---- Step 2: Train / val split (by sequence) ----------------------------
    rng = random.Random(cfg["seed"])
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_val = max(1, int(len(data) * cfg["eval_split"]))
    val_data = [data[i] for i in indices[:n_val]]
    train_data = [data[i] for i in indices[n_val:]]
    logger.info("Split: %d train, %d val sequences", len(train_data), len(val_data))

    # ---- Step 3: Load tokenizer ---------------------------------------------
    logger.info("Loading tokenizer: %s", cfg["model"])
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"], trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Step 4: Build datasets ---------------------------------------------
    ds_kwargs = dict(
        tokenizer=tokenizer,
        max_context_ops=cfg["max_context_ops"],
        max_target_ops=cfg.get("max_target_ops", None),
        example_stride=cfg["example_stride"],
        max_seq_len=cfg["max_seq_len"],
        # Preprocessing (applied on-the-fly in __getitem__)
        enable_context_shortening=shortening.get("enabled", True),
        context_shortening_max_chars=shortening.get("max_chars", 32),
        context_shortening_corner_cells_dim=shortening.get("corner_cells_dim", 2),
        remove_sheet_name=prep.get("remove_sheet_name", True),
        # Prompt
        include_sheet_name=cfg.get("include_sheet_name", False),
        prompt_template=cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE),
    )
    train_ds = OperationSequenceDataset(train_data, **ds_kwargs)
    val_ds = OperationSequenceDataset(val_data, **ds_kwargs)

    # ---- Step 5: Load model -------------------------------------------------
    logger.info("Loading model: %s", cfg["model"])
    model_kwargs: Dict[str, Any] = dict(trust_remote_code=True)
    if cfg["bf16"]:
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        # fp16 AMP requires fp32 model weights (GradScaler operates on fp32).
        # Many HF models default to bf16 in their config — force fp32 here.
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(cfg["model"], **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id

    # ---- Step 6: Optional LoRA ----------------------------------------------
    if cfg["use_lora"]:
        logger.info(
            "Applying LoRA: rank=%d, alpha=%d, dropout=%.2f",
            cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"],
        )
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            raise ImportError("LoRA requires `peft`. Install: pip install peft")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["lora_target_modules"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Required for gradient checkpointing + LoRA: frozen base model
        # layers don't produce gradients, so inputs need requires_grad=True
        if cfg["gradient_checkpointing"]:
            model.enable_input_require_grads()

    # ---- Step 7: Logging setup ----------------------------------------------
    report_to_list = []
    report_to_cfg = cfg.get("report_to", "tensorboard")
    if report_to_cfg == "all":
        report_to_list = ["tensorboard"]
        if cfg.get("wandb_project"):
            report_to_list.append("wandb")
    elif report_to_cfg == "none":
        report_to_list = ["none"]
    else:
        report_to_list = [report_to_cfg]

    if "wandb" in report_to_list and cfg.get("wandb_project"):
        os.environ["WANDB_PROJECT"] = cfg["wandb_project"]
        if cfg.get("wandb_run_name"):
            os.environ["WANDB_NAME"] = cfg["wandb_run_name"]

    run_name = cfg.get("wandb_run_name") or f"{cfg['model'].split('/')[-1]}_ft"

    # ---- Step 8: Training arguments -----------------------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        # Training
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["lr"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        # Precision
        fp16=cfg["fp16"] and not cfg["bf16"],
        bf16=cfg["bf16"],
        # Memory
        gradient_checkpointing=cfg["gradient_checkpointing"],
        dataloader_num_workers=cfg["dataloader_num_workers"],
        # Throughput: bucket similar-length examples to reduce padding
        group_by_length=cfg.get("group_by_length", False),
        # Logging & saving
        logging_dir=str(output_dir / "tb_logs"),  # TensorBoard log dir
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg["save_total_limit"],
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg.get("eval_steps", None),
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=False,
        report_to=report_to_list,
        # Misc
        seed=cfg["seed"],
        remove_unused_columns=False,
    )

    # ---- Step 9: Train ------------------------------------------------------
    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)

    # ---- Optional: Prediction accuracy callback ----------------------------
    callbacks = []
    if cfg.get("eval_test_cases"):
        from eval_callback import PredictionAccuracyCallback, load_test_cases

        test_cases_path = _PROJECT_ROOT / cfg["eval_test_cases"]
        if test_cases_path.exists():
            test_cases = load_test_cases(test_cases_path)
            accuracy_cb = PredictionAccuracyCallback(
                test_cases=test_cases,
                tokenizer=tokenizer,
                prompt_template=cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE),
                preprocessing=prep,
                eval_before_training=cfg.get("eval_before_training", True),
            )
            callbacks.append(accuracy_cb)
            logger.info(
                "Prediction accuracy eval enabled: %d test cases from %s",
                len(test_cases), test_cases_path,
            )
        else:
            logger.warning(
                "eval_test_cases path not found: %s", test_cases_path
            )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks if callbacks else None,
    )

    # ---- Step 9b: Resume from checkpoint if requested ----------------------
    resume_ckpt = None
    if cfg.get("resume"):
        # Find the latest checkpoint by step number (mtime is unreliable on shared FS)
        ckpts = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if ckpts:
            resume_ckpt = str(ckpts[-1])
            logger.info("Resuming from checkpoint: %s", resume_ckpt)
        else:
            logger.info("--resume requested but no checkpoints found; starting fresh.")

    logger.info("Starting training …")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ---- Step 10: Save final model + tokenizer ------------------------------
    final_dir = output_dir / "final"
    logger.info("Saving final model to %s", final_dir)

    # For LoRA: merge adapter weights into base model so the result is a
    # standalone model loadable by vLLM without peft.
    if cfg["use_lora"]:
        logger.info("Merging LoRA adapter into base model…")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(final_dir))
    else:
        trainer.save_model(str(final_dir))

    tokenizer.save_pretrained(str(final_dir))

    logger.info("Training complete ✓")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Finetune a causal LM on spreadsheet operation sequences.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_FINETUNING_ROOT / "configs" / "default.yaml",
        help="Path to YAML config file.",
    )
    # CLI overrides (None = use config value)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--processed_data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_context_ops", type=int, default=None)
    parser.add_argument("--example_stride", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_percentile", type=float, default=None)
    parser.add_argument("--use_lora", action="store_true", default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--bf16", action="store_true", default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "all", "none"],
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the latest checkpoint in output_dir.",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    if args.resume:
        cfg["resume"] = True

    # ---- Set up logging: console + file in output_dir -----------------------
    output_dir = _PROJECT_ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train.log"

    log_fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_fmt))
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to %s", log_file)

    # Log effective config (omit bulky fields)
    log_cfg = {
        k: v
        for k, v in cfg.items()
        if k not in ("lora_target_modules", "prompt_template")
    }
    logger.info("Effective config: %s", log_cfg)

    train(cfg)


if __name__ == "__main__":
    main()
