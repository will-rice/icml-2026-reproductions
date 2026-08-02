# DR Tulu Measurements and Pins

This page exposes concrete observations from `evidence/dr_tulu_results.json`.
Paper-reported benchmark values are listed only as located claims; they are not
reported as reproduced measurements.

## Attempt

- Attempt ID: `cd23c17f-37ff-4fb6-bda1-edd5d13d1f98`
- Paper ID: `97NEP1pyS3`
- Snapshot ID: `268793ddded7dcf5121263482ebbabd087edaf6350f1f0f1126541d144e172ac`
- arXiv artifact: `2511.19399`
- Upstream repository: `https://github.com/rlresearch/dr-tulu`
- Upstream commit: `9d7b0371c085e9311ddec483ed39768c0bd9fe99`

## Artifact Sources

- Model card: `evidence/sources/DR-Tulu-8B/README.md`
- No-RLER model card: `evidence/sources/DR-Tulu-No-RLER-8B/README.md`
- RL data card: `evidence/sources/dr-tulu-rl-data/README.md`
- Repository script: `rl/open-instruct/train_dr_tulu.sh`
- Trainer source: `rl/open-instruct/open_instruct/grpo_fast.py`
- Rubric utilities: `rl/open-instruct/open_instruct/search_rewards/utils/rubric_utils.py`
- Agent client: `agent/dr_agent/client.py`

## Claim Checks

- Claim 1 status: `verified`
- Claim 1 max active rubrics located in code: `5`
- Claim 1 generated-rubric append located: `true`
- Claim 1 score-standard-deviation pruning located: `true`
- Claim 2 status: `verified`
- Claim 2 base model located: `rl-research/DR-Tulu-SFT-8B`
- Claim 2 RL data located: `rl-research/dr-tulu-rl-data`
- Claim 2 MCP tools located: `snippet_search`, `google_search`, `browse_webpage`
- Claim 3 status: `inconclusive`
- Claim 3 reported average located but unrecomputed: `65.6`
- Claim 3 reported prior-open-baseline margin located but unrecomputed: `15.6`
- Claim 4 status: `inconclusive`
- Claim 4 reported cost ratio located but unrecomputed: `1000x`
- Claim 5 status: `inconclusive`
- Claim 5 No-RLER ablation card located: `true`
- Claim 6 status: `inconclusive`
- Claim 6 Qwen3-8B judge/rubric-generator mention located: `true`

## Limits

- GPU training runs performed: `0`
- ScholarQA-CSv2 evaluations recomputed: `0`
- HealthBench evaluations recomputed: `0`
- ResearchQA evaluations recomputed: `0`
- DeepResearchBench evaluations recomputed: `0`
- Paid API judge or search calls performed: `0`
- Metered API cost: `$0.00`
- Paper-reported table values copied as reproduced measurements: `0`
