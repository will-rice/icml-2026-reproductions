# Claim 4: GPT-5 R results


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ebfe8bb03b5e", "created_at": "2026-07-21T16:05:48+00:00", "title": "Replication status"}
-->
**NOT REPLICATED.** The paper values under test are **32.7 / 29.4 / 41.6**. They are recorded here only to identify the claim; **no verdict is offered**.

The paper's full row is GPT-5-R single-action repredict: **32.7 UAS, 29.4 AR, 41.6 precision, and 24.8 pCov**, with stride 1, context 32, the paper's greedy acceptance rule, and reasoning effort set to low.

## Future rerun contract

Prerequisites are the pinned NAPE checkout at `ac0d10e4dc345f982a5665a2c4bdb6b752d663f2`, its 52 files under `data/trajectories`, an OpenAI provider deployment whose model ID is `gpt-5`, `OPENAI_API_KEY`, outbound network access, and paid inference budget. No original GPT-5-R responses or response cache were released. In addition, the pinned OpenAI runner does not consume `solver.cache_enabled` or `solver.cache_path`, so this path requires live calls rather than a cache replay.

The exact closest pinned entrypoint is runnable after replacing the placeholder model:

```bash
uv sync --all-packages --extra openai
cd external/NAPE
uv run python -c 'from pathlib import Path; import yaml; source=Path("configs/evaluation/base_run_repredict.yaml"); target=Path("/tmp/nape-claim-4.yaml"); config=yaml.safe_load(source.read_text()); config["name"]="claim_4_gpt5r_closest"; config["output_dir"]="results/claim_4_gpt5r_closest"; config["solver"]["model"]="gpt-5"; target.write_text(yaml.safe_dump(config, sort_keys=False))'
OPENAI_API_KEY="$OPENAI_API_KEY" uv run --package next-action-pred-eval --extra openai python scripts/run_batch_evaluation.py --config /tmp/nape-claim-4.yaml --no-resume
```

**Pinned implementation gap:** this command is a GPT-5 non-reasoning run, not GPT-5-R. The pinned `ExperimentConfig`, `ChatSolver`, and batch runner expose no `reasoning_effort` field and never pass `reasoning_effort="low"` to the provider. Exact GPT-5-R reproduction therefore has no command at this revision; it first requires a source change carrying that field from YAML through `ChatSolver` to `OpenAIAdapter.chat_with_response`, followed by the same batch command.

The run writes `results/claim_4_gpt5r_closest/batch_summary.csv` and `experiment_summary.json`. Compare `uas_pct`, `acceptance_rate`, `avg_precision`, and `coverage_pct_tp` against the four paper values; the aggregate JSON reports mean UAS, acceptance rate, and precision, while pCov remains in the per-trajectory CSV. The named model outputs, credentials, cache inputs, and paid budget were unavailable here, so no proxy result is reported. The machine-readable status remains in `claims_4_6_status.json`.
