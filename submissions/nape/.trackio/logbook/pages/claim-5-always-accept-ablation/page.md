# Claim 5: Always-accept ablation


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f375a38422d8", "created_at": "2026-07-21T16:05:57+00:00", "title": "Replication status"}
-->
**NOT REPLICATED.** The paper ablation under test reports **-19.2 UAS** for always accepting predictions. It is recorded here only to identify the claim; **no verdict is offered**.

The full paper row is multi-action GPT-5 with stride 1, context 32, unlimited operations per call, and the ALWAYS rule: **-19.2 UAS, 100.0 AR, 9.3 precision, and 8.1 pCov**. The pinned YAML name for ALWAYS is `accept_all`.

## Future rerun contract

Prerequisites are the pinned NAPE checkout at `ac0d10e4dc345f982a5665a2c4bdb6b752d663f2`, its 52 released trajectories, OpenAI model ID `gpt-5` in non-reasoning chat mode, `OPENAI_API_KEY`, outbound network access, and paid inference budget. No original responses or replay cache were released, and the pinned OpenAI runner ignores the YAML response-cache fields, so live provider calls are required.

The pinned heuristic sweep implements this protocol. Replace only the model placeholder and output identity, then run all paper heuristics including `accept_all`:

```bash
uv sync --all-packages --extra openai
cd external/NAPE
uv run python -c 'from pathlib import Path; import yaml; source=Path("configs/evaluation/heuristic_sweep.yaml"); target=Path("/tmp/nape-claim-5.yaml"); config=yaml.safe_load(source.read_text()); config["name"]="claim_5_heuristic_sweep"; config["output_dir"]="results/claim_5_heuristic_sweep"; config["solver"]["model"]="gpt-5"; target.write_text(yaml.safe_dump(config, sort_keys=False))'
OPENAI_API_KEY="$OPENAI_API_KEY" uv run --package next-action-pred-eval --extra openai python scripts/run_sweep.py /tmp/nape-claim-5.yaml --no-resume
```

Inspect the `accept_all` variant in `results/claim_5_heuristic_sweep/`, plus `sweep_summary.csv`. Compare `uas_pct`, `acceptance_rate`, `avg_precision`, and `coverage_pct_tp` with **-19.2 / 100.0 / 9.3 / 8.1**. The paper also reports that 51/52 trajectories hit the 120%-of-ground-truth user-step cap; verify `user_step_limit_reached` in `batch_summary.csv`.

The named outputs, credentials, cache inputs, and paid budget were unavailable here. No proxy comparison was made, and the machine-readable status remains in `claims_4_6_status.json`.
