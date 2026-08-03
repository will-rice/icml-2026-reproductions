# Efficient, Property-Aligned Fan-Out Retrieval via RL-Amortized Diffusion

- Attempt: `8a83f44b-e3db-4c2b-acf7-d233a750fdcc`
- Paper: `4P9cEcinYP`
- Snapshot: `34237d5702ab85038fbe25e4409a2115b90ef0257ab437d9511f2d66ded5fdd5`
- Source pin: `arxiv:2603.06397@sha256:3602626196bb2747970029de2a6f9b8086e4450a8f67c105d2954911a1d8a568`

## Claim Evidence

### Claim 1: toy

R4T trains a fan-out language model with set-level rewards, synthesizes objective-consistent query-set supervision, and trains a diffusion retriever for single-pass fan-out in embedding space (Figure 1).

The pinned arXiv source describes the three R4T stages and this package exercises a tiny set-level reward plus synthetic target compilation. No released FOLM or diffusion training code is available.

### Claim 2: inconclusive

R4T consistently outperforms fan-out baselines across Open-Ended Abstract Retrieval datasets and metrics (Table 1).

task_1_result.tex contains paper-reported OAR table rows, but no raw Polyvore/Music inputs, model outputs, or evaluation scripts are released for recomputation.

### Claim 3: inconclusive

R4T improves weakly supervised compositional retrieval on Polyvore relative to no-fan-out, zero-shot fan-out, and best-of-N baselines (Table 2).

task_2_result.tex contains paper-reported WSCR values, but no Polyvore preprocessing, broad-query generation outputs, or retrieval outputs are released.

### Claim 4: toy

Jointly optimizing groundedness, alignment, and diversity prevents reward-collapse behavior during fan-out LM training (Figure 4).

The source includes reward-collapse figure assets and text. The toy reward check gives collapsed total 0.7 and diverse total 0.706904, showing diversity/alignment terms can penalize collapse on a synthetic fixture.

### Claim 5: inconclusive

The diffusion fan-out retriever maintains sub-second latency for small batches and reaches order-of-magnitude speedups over autoregressive LLM fan-out at larger batch sizes (Figure 5).

query_fanout_efficiency.pdf and manuscript text report latency, but no executable diffusion retriever, autoregressive baseline, hardware harness, or raw latency log is released.

## Audits

```json
{
  "method_text_markers": {
    "latency_text": true,
    "reward_components": true,
    "three_stage_method": true
  },
  "source_manifest": {
    "dataset_files": [],
    "file_count": 22,
    "has_latency_figure": true,
    "has_main_tex": true,
    "has_reward_figures": true,
    "has_table1_tex": true,
    "has_table2_tex": true,
    "model_files": [],
    "python_files": []
  },
  "tables": {
    "table_values_are_recomputed": false,
    "task1_has_best_of_n": true,
    "task1_r4t_mentions": 5,
    "task2_has_polyvore_metrics": true,
    "task2_r4t_mentions": 4
  },
  "toy_reward": {
    "collapsed": {
      "alignment": 1.0,
      "diversity": 0.0,
      "groundedness": 1.0,
      "total": 0.7
    },
    "diverse": {
      "alignment": 0.569036,
      "diversity": 0.528595,
      "groundedness": 0.902369,
      "total": 0.706904
    },
    "supervision": [
      {
        "query": "bohemian festival style",
        "target_centroid": [
          0.566667,
          0.566667
        ],
        "target_count": 3
      }
    ]
  }
}
```

No paper-reported table or latency value is presented as a recomputed measurement.
