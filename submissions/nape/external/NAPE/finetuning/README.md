# Finetuning — Causal LM on Spreadsheet Operation Sequences

Fine-tune small causal language models (Qwen-0.5B, SmolLM2-360M, etc.) on
~10K spreadsheet operation sequences using standard **next-token prediction**.

## Quick Start

```bash
# From the project root: next_action_pred_eval/

# Step 1 — Export raw sequences as JSONL (one-time, ~30 seconds)
python finetuning/data_preparation.py

# Step 2 — Train (auto-runs Step 1 if JSONL doesn't exist)
python finetuning/train.py --config finetuning/configs/default.yaml
```

## Data Pipeline

```
<training-data>/        data_preparation.py         dataset.py (__getitem__)
  ~N folders     ─────────────────►  JSONL    ─────────────────►  Tokenised
  framework_output.txt   (raw ops,    file      (preprocess +       examples
                         p95 filter)            format + tokenise)
```

**Key design**: the JSONL file contains **raw operations only** — no
preprocessing baked in. Preprocessing (value shortening, sheet-name
stripping) happens **on-the-fly** in the dataloader. This means:

- Changing a preprocessing param in the config → immediate effect, no re-export
- The JSONL only needs re-generation if the source data changes
- The file is human-readable: `head -3 finetuning/processed_data/sequences.jsonl`

## Preview Training Data

Inspect sample training examples before committing to a full run:

```bash
# Show 5 examples with token-count statistics
python finetuning/data_preparation.py \
    --preview 5 \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --max_context_ops 128 \
    --example_stride 64
```

This prints:
- Dataset statistics (sequences, ops, estimated examples)
- Sample formatted prompts (exactly as the model will see them)
- Token length distribution (min / p25 / median / p75 / max)

## CLI Overrides

Any config value can be overridden on the command line:

```bash
python finetuning/train.py \
    --config finetuning/configs/default.yaml \
    --model HuggingFaceTB/SmolLM2-360M \
    --max_context_ops 64 \
    --example_stride 32 \
    --epochs 5 \
    --lr 3e-5 \
    --output_dir finetuning/results/smollm2-360m
```

## Configurable Prompt Template

The prompt template is set in the config YAML and must contain an
`{actions}` placeholder where the numbered action list is inserted.

Default (matches `CompletionSolver`):
```
Complete the sequence of actions to build the following spreadsheet by identifying and extending key patterns.

{actions}
```

To experiment with a different prompt, edit `prompt_template` in the config:

```yaml
prompt_template: >-
  You are a spreadsheet assistant. Continue the following operations:

  {actions}
```

## TensorBoard

Training metrics are logged to `{output_dir}/tb_logs/` by default:

```bash
tensorboard --logdir finetuning/results/tb_logs
```

To use Weights & Biases instead (or in addition):

```bash
python finetuning/train.py --report_to all --wandb_project spreadsheet-ft
```

## Architecture: Why `transformers.Trainer` (not TRL)

We use `transformers.Trainer` rather than `trl.SFTTrainer` because:

- **Standard causal LM objective** — loss on all tokens, no prompt masking
  needed. `Trainer` handles this natively.
- **Simpler, more transparent** — fewer abstractions, easier to debug and
  extend (e.g., for custom logit-based stop decoding later).
- **TRL would add value if** we later want: prompt masking (loss only on
  continuation tokens), sequence packing (for shorter examples), or
  RLHF/DPO training. These can be adopted incrementally.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_context_ops` | 128 | Operations per training example (matches solver) |
| `example_stride` | 64 | Step between windows; 64 = 50% overlap |
| `max_seq_len` | 2048 | Hard token limit per example |
| `max_percentile` | 95.0 | Filter out sequences with more ops than this percentile |
| `preprocessing.context_shortening.max_chars` | 32 | Match `base_run.yaml` |
| `preprocessing.context_shortening.corner_cells_dim` | 2 | Match `base_run.yaml` |
| `prompt_template` | (see above) | Configurable prompt with `{actions}` placeholder |
| `include_sheet_name` | false | Experimental: prepend "Active sheet: ..." |

### Choosing `example_stride`

| Stride | Overlap | Examples (~10K seqs) | Notes |
|--------|---------|----------------------|-------|
| 128 (= max_context_ops) | 0% | ~120K | Fastest epochs, no overlap |
| 64 | 50% | ~240K | Good balance (default) |
| 32 | 75% | ~480K | More data augmentation |
| 1 | ~99% | ~15M | Very slow, maximum coverage |

## Inference Integration

Point the evaluation config at the finetuned checkpoint:

```yaml
# configs/evaluation/local_model_run.yaml
solver:
  type: llm
  adapter: local
  model: finetuning/results/final   # ← finetuned checkpoint path
  temperature: 0.0
  max_tokens: 4096
  remove_sheet_name: true
```

The `CompletionSolver` → `LocalModelAdapter.complete()` pipeline works
unchanged because training data is formatted **byte-identically** to the
solver's completion prompt (when using the default prompt template).

### Stop-decoding strategy

The model generates until EOS or `max_tokens`. A custom logit/logprob-based
decoding strategy for determining when to stop suggesting operations can be
layered on later — the model has learned the distribution of operation
sequences and its per-token confidence can be inspected at decode time.

## Train / Inference Alignment

| Step | Solver code | Training code |
|------|-------------|---------------|
| 1. Shorten values | `shorten_symbolic_values(max_chars=32, corner_dim=2)` | Same function, same params (in dataloader) |
| 2. Strip sheet names | `compress_symbolic(remove_sheet_name=True)` | Same function (in dataloader) |
| 3. Number actions | `_format_actions` → `"1. OP \| R \| V"` | Same logic (1-indexed, in dataloader) |
| 4. Prompt header | `DEFAULT_COMPLETION_TEMPLATE` | Configurable `prompt_template` (default matches) |
| 5. Tokenise | `tokenizer(prompt)` — default BOS, no chat template | Same tokenizer, same defaults |

## Directory Structure

```
finetuning/
├── configs/
│   └── default.yaml          # all hyperparameters
├── results/                   # checkpoints, tb_logs/ (gitignored)
├── processed_data/
│   └── sequences.jsonl        # raw ops, human-readable (gitignored)
├── data_preparation.py        # export raw data + preview
├── dataset.py                 # PyTorch Dataset + collator (on-the-fly preprocessing)
├── train.py                   # training entrypoint
└── README.md                  # this file
```

## JSONL Format

Each line in `sequences.jsonl` is a JSON object:

```json
{"id": "sheet_042", "ops": ["SetValue | A1 Sheet1 | Hello", "SetValue | B1 Sheet1 | =A1+1", ...]}
```

- `id`: folder name from the source training-data directory
- `ops`: raw operation strings (no preprocessing applied)

Inspect with standard tools:
```bash
# Count sequences
wc -l finetuning/processed_data/sequences.jsonl

# View first 3
head -3 finetuning/processed_data/sequences.jsonl | python -m json.tool

# Find a specific sheet
grep "sheet_042" finetuning/processed_data/sequences.jsonl
```

## LoRA (optional)

For larger models or limited VRAM:

```bash
python finetuning/train.py --use_lora --lora_rank 16
```

For ≤1B models, full finetune is recommended (the default).
