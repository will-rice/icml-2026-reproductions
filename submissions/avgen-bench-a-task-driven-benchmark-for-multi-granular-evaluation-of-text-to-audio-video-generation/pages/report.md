# AVGen-Bench Evidence Report

- Attempt: `dca81643-55fa-4d3c-9244-d3d1a00119aa`
- Paper: `aJdgt8xDMy`
- Snapshot: `41692f328d154e4fad790fb8c89aa276452ce49b8aaa18064abb9c47a897d622`

## Claim Results

| # | Status | Evidence summary |
|---:|---|---|
| 1 | verified | Released prompts contain 235 prompts across 11 category files; scoring code exposes 3 evaluation groups. Limitation: The raw prompt JSON verifies subcategory coverage; the domain wording is inferred from released benchmark/scoring structure. |
| 2 | verified | Aggregate code defines 11 dimensions across 3 groups and released source exposes 11 expected evaluator modules. Limitation: This verifies metric/module availability and score formula, not full model evaluation reruns. |
| 3 | toy | The release includes a benchmark-comparison asset and 11 metric dimensions; prompt mean length is 70.2 words. Limitation: No independent rerun or audit of prior benchmark prompt complexity was performed. |
| 4 | toy | Parsed 13 released leaderboard rows; high-basic/low-fine models: ['Veo 3.1-fast', 'Veo 3.1-quality', 'Wan2.6', 'Seedance-1.5 Pro']. Limitation: The evidence recomputes patterns from released aggregate README values, not raw evaluator outputs. |
| 5 | inconclusive | Human-correlation artifacts found: []. Limitation: Expert human judgment data are required to recompute correlations and were not found in released artifacts. |
| 6 | toy | Stability script present: True; repeat output artifacts found: []. Limitation: Prompt-subset/repeated-run cached outputs are required for numeric stability reproduction and were not found. |
