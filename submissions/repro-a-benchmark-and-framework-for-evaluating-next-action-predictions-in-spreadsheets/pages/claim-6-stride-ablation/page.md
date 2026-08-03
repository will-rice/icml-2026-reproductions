# Claim 6: Stride ablation


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_61b50c042ed8", "created_at": "2026-07-21T16:06:12+00:00", "title": "Replication status"}
-->
**NOT REPLICATED.** The paper stride result under test changes from **27.4 to 10.6**. It is recorded here only to identify the claim; **no verdict is offered**.

This is the paper's **single-action repredict** GPT-5 ablation with context 32, greedy acceptance, and stride values 1, 2, 4, and 8. The reported UAS values are **27.4, 22.6, 16.8, and 10.6**; the same rows also report AR, precision, and pCov.

## Future rerun contract

Prerequisites are the pinned NAPE checkout at `ac0d10e4dc345f982a5665a2c4bdb6b752d663f2`, all 52 trajectories, OpenAI model ID `gpt-5` in non-reasoning chat mode, `OPENAI_API_KEY`, outbound network access, and paid inference budget. No original model outputs or replay cache were released. The pinned OpenAI runner does not use its configured response-cache path, so this is a live rerun.

**Pinned configuration gap:** `configs/evaluation/hyperparameter_ablation.yaml` is directly runnable, but it encodes the paper's multi-action Table 6 rather than the 27.4-to-10.6 single-action Table 3 claim. The closest pinned entrypoint is still `scripts/run_sweep.py`; make these explicit supported edits: enable `repredict_after_accept`, set `max_predictions_per_step` and `solver.num_op_to_pred` to 1, retain `steps_saved`, and restrict `sweep_independent` to stride 1/2/4/8.

The following command performs exactly those edits before invoking the pinned entrypoint:

```bash
uv sync --all-packages --extra openai
cd external/NAPE
uv run python -c 'from pathlib import Path; import yaml; source=Path("configs/evaluation/hyperparameter_ablation.yaml"); target=Path("/tmp/nape-claim-6.yaml"); config=yaml.safe_load(source.read_text()); config["name"]="claim_6_single_action_stride"; config["output_dir"]="results/claim_6_single_action_stride"; config["repredict_after_accept"]=True; config["max_predictions_per_step"]=1; config["solver"]["model"]="gpt-5"; config["solver"]["num_op_to_pred"]=1; config["sweep_independent"]={"stride.interval":[1,2,4,8]}; target.write_text(yaml.safe_dump(config, sort_keys=False))'
OPENAI_API_KEY="$OPENAI_API_KEY" uv run --package next-action-pred-eval --extra openai python scripts/run_sweep.py /tmp/nape-claim-6.yaml --no-resume
```

Inspect `results/claim_6_single_action_stride/sweep_summary.csv` and each variant's `batch_summary.csv`/`experiment_summary.json`. Compare `uas_pct`, `acceptance_rate`, `avg_precision`, and `coverage_pct_tp`; the paper rows are stride 1 **27.4 / 30.9 / 44.8 / 20.7**, stride 2 **22.6 / 36.5 / 48.4 / 15.9**, stride 4 **16.8 / 42.3 / 53.2 / 9.4**, and stride 8 **10.6 / 43.7 / 55.1 / 7.1**.

The named outputs, credentials, cache inputs, and paid budget were unavailable here. No proxy comparison was made, and the machine-readable status remains in `claims_4_6_status.json`.
