# next_action_pred_eval

**[🌐 Project page & live demo](https://napeval.github.io)** &nbsp;·&nbsp; **[📄 Paper (arXiv)](https://arxiv.org/abs/2606.13802)**

Benchmark and evaluation framework for **predictive auto-completion in spreadsheets**. Given a user's editing history (a sequence of symbolic cell operations), the framework asks a solver to predict the next operation(s) the user is likely to perform, and grades the prediction by comparing the resulting workbook **state** against ground truth — not the raw operation strings.

## Why state-level evaluation?

Two different operation sequences can produce the same workbook. A purely string-based metric would unfairly penalise an accurate but stylistically different prediction. The orchestrator therefore applies the predicted ops, applies the ground-truth ops, and compares the two workbook states cell-by-cell.

## Install

Requires **Python 3.10+**.

```bash
# Minimal install
pip install -e .

# With OpenAI adapter
pip install -e ".[openai]"

# With local HuggingFace models
pip install -e ".[local]"

# Streamlit dashboard
pip install -e ".[dashboard]"

# Everything
pip install -e ".[all]"

# Dev / test tooling
pip install -e ".[dev]"
```

## Quick start

```bash
# Single config
python scripts/run_evaluation.py --config configs/evaluation/base_run.yaml

# Batch (parallel, with resume + per-trajectory caching)
python scripts/run_batch_evaluation.py --config configs/evaluation/model_sweep.yaml

# Sweep — cross-product of parameter values
python scripts/run_sweep.py configs/evaluation/sweep_example.yaml

# Streamlit results dashboard
python scripts/run_dashboard.py
```

## The symbolic DSL

Every spreadsheet edit is encoded as a pipe-delimited string:

```
OPERATION_TYPE | Sheet!Range | value
```

Examples:

| Symbolic string | Meaning |
|---|---|
| `VALUE \| Sheet1!A1 \| 42` | Set A1 to the literal value 42 |
| `FORMULA \| Sheet1!C2 \| =SUM(A2:B2)` | Set C2 to a formula |
| `FONT_BOLD \| Sheet1!A5 \| True` | Bold A5 |
| `BORDER_TOP \| Sheet1!A8:G8 \| Thin, Continuous, #000000` | Top border on a row |
| `INPUT \| Sheet1!A1:B2 \| [[1,2],[3,4]]` | Bulk-input a 2×2 block |

The canonical mapping lives in [`src/next_action_pred_eval/core/operations/__init__.py`](src/next_action_pred_eval/core/operations/__init__.py).

## Data

The repository ships **52 trajectories** under `data/trajectories/` plus the corresponding raw workbooks under `data/raw/<id>/` (each containing `operations.txt`, `predictable_state.json`, `sheet_image.png`, `spreadsheet.xlsx`).

Each `data/trajectories/<id>.json` looks like:

```json
{
  "name": "0000afae",
  "operations": [
    "MERGE | Sheet1!A1:G1 | true",
    "INPUT | Sheet1!A1:A2 | [[\"Title\"], [\"www.example.com\"]]",
    "FONT_BOLD | Sheet1!A5 | True"
  ]
}
```

## Solvers

Built-in solvers (`src/next_action_pred_eval/evaluation/`). Any class that
implements the `ISolver` interface
([`evaluation/solver.py`](src/next_action_pred_eval/evaluation/solver.py))
can be evaluated. The repo ships three families:

**LLM solvers** (`evaluation/baselines/`):

| Solver | `type` | What it does |
|---|---|---|
| `ConstantSolver` | `constant` | Predicts nothing — a no-skill baseline |
| `ChatSolver` | `chat` (or `llm`) | Calls an LLM in chat mode with a system + user prompt |
| `CompletionSolver` | `completion` | Calls an LLM in completion mode (raw prompt → continuation) |

**Classical sequence solvers** (non-LLM, trained on operation sequences):

| Solver | `type` | What it does |
|---|---|---|
| `NGramSolver` | `ngram` | Back-off n-gram frequency model |
| `OnlineNGramSolver` | `online_ngram` | Suffix-match model that learns within a single trajectory (no training) |
| `LSTMSolver` | `lstm` | Small GRU sequence model (needs `torch`) |
| `XGBoostSolver` | `xgboost` | Gradient-boosted next-op predictor (needs `xgboost`) |

The classical solvers share a common featurizer
(`evaluation/baselines/featurizer.py`) and decoding loop. Training scripts
for the LSTM and XGBoost models live in
[`examples/baselines/`](examples/baselines/). The n-gram and online n-gram
solvers need no separate training step.

Select a solver in YAML via the `solver.type` field — see
[`configs/evaluation/base_run_repredict.yaml`](configs/evaluation/base_run_repredict.yaml)
for the single-action protocol used with the smaller solvers.

### Plugging in your own LLM provider

The framework targets any class that implements the
[`LLMAdapter`](src/next_action_pred_eval/utils/llm/base.py) interface.
Built-in adapters cover OpenAI-compatible APIs and HuggingFace local
models. To plug in anything else (Anthropic, Cohere, Azure, an internal
endpoint, etc.), write a subclass and load it via the generic
`custom` adapter type in YAML:

```yaml
solver:
  type: llm
  adapter: custom
  adapter_class: my_package.my_module.MyAdapter
  adapter_kwargs:
    endpoint: https://my-endpoint/v1
    timeout: 30
```

`create_adapter("custom", adapter_class=..., adapter_kwargs={...})`
imports the dotted path, verifies it inherits from `LLMAdapter`, and
instantiates it.

## Configuration

YAML configs in `configs/evaluation/` define experiments. See
[`configs/README.md`](configs/README.md) for a description of each
included config. Sweeps support three modes:

* `sweep:` — cross-product of parameter lists
* `sweep_zip:` — lockstep pairing
* `sweep_independent:` — one-at-a-time variation

## Repository layout

```
src/next_action_pred_eval/
├── core/             # operations, symbolic DSL, state builder, transforms
├── evaluation/       # orchestrator, solver interface, metrics, acceptance heuristics
│   └── baselines/    # ChatSolver, CompletionSolver, prompts, featurizer
├── generation/       # training-data generation pipeline (sampling, regions, sequencing, refinement)
├── dashboard/        # Streamlit results dashboard
└── utils/
    ├── llm/          # LLMAdapter ABC + OpenAIAdapter, LocalModelAdapter, create_adapter()
    ├── codegen/      # convert Operations into OfficeJS / openpyxl / xlwings code
    ├── workbook/     # workbook state utilities
    └── image_utils/  # screenshot + image helpers

configs/              # YAML experiment configs (see configs/README.md)
scripts/              # CLI entry points
data/
├── trajectories/     # 52 symbolic trajectories (JSON)
└── raw/              # per-trajectory raw artifacts (.xlsx, .png, predictable_state.json)
tests/                # pytest suite
finetuning/           # causal-LM fine-tuning on operation sequences
examples/baselines/   # non-LLM baselines (LSTM, XGBoost)
```

## Running tests

```bash
pytest tests/ -q
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE). Free for research,
educational, and other noncommercial use. Contact the authors for
commercial licensing.

## Citation

If you use this benchmark or framework, please cite:

```bibtex
@inproceedings{agrawal2026nape,
  title     = {A Benchmark and Framework for Evaluating Next Action Predictions in Spreadsheets},
  author    = {Agrawal, Tejas and Le, Vu and Gulwani, Sumit and Verbruggen, Gust},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

Preprint: [arXiv:2606.13802](https://arxiv.org/abs/2606.13802)
