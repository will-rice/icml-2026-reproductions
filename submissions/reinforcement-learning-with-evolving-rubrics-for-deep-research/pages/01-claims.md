# Claims and Evidence Analysis

This reproduction audit evaluated 6 primary claims from the paper *Reinforcement Learning with Evolving Rubrics for Deep Research* (paper ID: `97NEP1pyS3`).

## Summary of Audit Findings

| Claim # | Claim Description | Status | Key Evidence / Observation |
| --- | --- | --- | --- |
| 1 | RLER maintains a fixed-size rubric buffer updated with rubrics from current rollouts and pruned by rollout-score variance (Figure 2) | Verified | Max active rubrics cap of 5 (`max_active_rubrics: 5`), adaptive rubric reward, score std-dev pruning verified in `rubric_utils.py` |
| 2 | DR Tulu-8B is trained as an open long-form deep-research agent using supervised cold start followed by online RL with async tool calls (Sections 4.2 & 4.3) | Verified | Model base `rl-research/DR-Tulu-SFT-8B`, dataset `rl-research/dr-tulu-rl-data`, async tools (`snippet_search`, `google_search`, `browse_webpage`) verified |
| 3 | Across ScholarQA-CSv2, HealthBench, ResearchQA, and DeepResearchBench, DR Tulu-8B RL averages 65.6 and exceeds best baseline by 15.6 points (Table 1) | Inconclusive | Model card contains reported table; primary evaluation reruns were not executed in this CPU audit |
| 4 | DR Tulu-8B lies on performance-cost Pareto frontier and is reported as about 1000x cheaper per ScholarQA-CSv2 query than OpenAI Deep Research (Figure 1) | Inconclusive | Metered API cost recorded as $0.00; paid search/judge APIs and full agent execution not rerun |
| 5 | Evolving rubrics outperform using only initial search-based rubrics during RL training, with gap widening over training (Figure 6) | Inconclusive | `DR-Tulu-No-RLER-8B` ablation card present; full training-curve artifacts not recomputed on CPU |
| 6 | Using Qwen3-8B as judge and rubric generator improves over SFT baseline, though underperforming GPT judging by 1.3 points (Table 4) | Inconclusive | Judge model configurable in codebase; Qwen3-8B judge ablation table not recomputed |

## Computational Evidence Details

1. **Rubric Buffer Mechanics**: Pinned code in `rl/open-instruct/open_instruct/search_rewards/utils/rubric_utils.py` confirms that `max_active_rubrics = 5`, active/inactive fields are tracked, and rubrics are pruned based on standard deviation of rollout scores.
2. **Tooling & Infrastructure**: Pinned code in `agent/dr_agent/client.py` and `rl/open-instruct/train_dr_tulu.sh` confirms MCP tool execution and async execution for web search and browsing.
