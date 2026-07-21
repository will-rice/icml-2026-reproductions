# Configurations

YAML configs that drive evaluation, generation, and finetuning runs.

## `evaluation/`

| Config | Purpose | Paper reference |
|---|---|---|
| `base_run.yaml` | Single full run, multi-action mode. Solver emits a variable-length block per call. Good first thing to run after install. | §5, Table 1 row |
| `base_run_repredict.yaml` | Single full run, single-action repredict mode. Solver emits at most one op per call; the loop re-invokes after each acceptance. Use this with small / classical solvers that can't decide when to stop. | §5, Table 4 row |
| `model_sweep.yaml` | Compare N models under identical settings via `sweep_zip`. | Table 1 / Table 4 |
| `heuristic_sweep.yaml` | Compare N acceptance heuristics via `sweep_independent`. | Table 2 |
| `hyperparameter_ablation.yaml` | Vary one hyperparameter at a time (stride, context, m, shortening). | Table 3 |
| `local_model_run.yaml` | Same protocol as `base_run_repredict` but uses a HuggingFace local model via `LocalModelAdapter`. | §5 |
| `sweep_example.yaml` | Minimal example documenting the `sweep:` cross-product syntax. | — |

All evaluation configs share the same schema defined in
[`experiment_config.py`](../src/next_action_pred_eval/evaluation/experiment_config.py).

### Paper defaults

Unless noted otherwise, the paper's default settings (used in `base_run.yaml`
and `base_run_repredict.yaml`) are:

| Setting | Value | YAML field |
|---|---|---|
| Stride (s) | 1 | `stride.interval` |
| Context (c) | 32 ops | `max_context_ops` |
| Max ops per call (m) | ∞ (multi) / 1 (single) | `solver.num_op_to_pred` |
| Context shortening | on | `context_shortening.enabled` |
| Heuristic | greedy | `heuristics: [steps_saved]` |
| Online mode | on | `online_mode` |

### Acceptance heuristic naming

The paper uses short names; the codebase uses more descriptive ones.

| Paper name | YAML value | Rule |
|---|---|---|
| greedy | `steps_saved` | ops saved ≥ 1 |
| hybrid-1 | `ideal_user` | precision ≥ 0.9 AND ops saved ≥ 1 |
| greedy-2 | `ssav2` | ops saved ≥ 2 |
| hybrid-2 | `ideal_user_strict` | precision = 1.0 AND ops saved ≥ 2 |
| p100 | `precision_100` | precision = 1.0 |
| p90 | `precision_90` | precision ≥ 0.9 |
| p60 | `precision_60` | precision ≥ 0.6 |
| always | `accept_all` | unconditionally accept |

> Note: there is also a code-level `HEURISTIC_GREEDY` constant whose rule
> is *ops saved ≥ 0* (i.e., the paper's "greedy" relaxed by one step).
> The configs in this repo use `steps_saved` to match the paper exactly.

## `generation/`

| Config | Purpose |
|---|---|
| `default.yaml` | Minimal sequencing-engine config used by the data-generation pipeline. |
| `pipeline.yaml` | Full pipeline config: targets, region analysis, sequencing, refinement. |
