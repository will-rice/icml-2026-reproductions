# R4T Reproduction Design

- Attempt: `8a83f44b-e3db-4c2b-acf7-d233a750fdcc`
- Owner: `codex-paper-owner-05`
- Fencing token: `1`
- Paper: `4P9cEcinYP`
- Snapshot: `34237d5702ab85038fbe25e4409a2115b90ef0257ab437d9511f2d66ded5fdd5`
- Title: `Efficient, Property-Aligned Fan-Out Retrieval via RL-Amortized Diffusion`

## Pinned Artifacts

- Paper page: `https://research.google/pubs/efficient-property-aligned-fan-out-retrieval-via-rl-compiled-diffusion-2/`
- Paper source: `arxiv:2603.06397`, e-print SHA-256 `3602626196bb2747970029de2a6f9b8086e4450a8f67c105d2954911a1d8a568`
- Paper PDF: `arxiv:2603.06397`, PDF SHA-256 `19b97f8264d22e28dbdced297d16d1686c68de2bbc0c756a2f9330fc023490bb`

Searches found no public GitHub repository, Hugging Face model, or Hugging
Face dataset for the exact paper title, arXiv ID, `R4T`, or
`Retrieve-for-Train`.

## Evidence Plan

Build a CPU-only evidence package in
`submissions/efficient-property-aligned-fan-out-retrieval-via-rl-amortized-diffusion`
with:

1. `generate_evidence.py` that downloads the pinned arXiv e-print, audits its
   file manifest, table TeX, method text, and figure assets, and writes
   `evidence/bundle.json` plus `pages/report.md`.
2. Unit tests written before implementation for source-manifest audit, Table
   1/Table 2 parsing, absence of executable artifacts, toy reward behavior,
   exact claim hashes, and expected statuses.
3. A small Gradio report app for served judge inspection.

## Claim Strategy

1. `fe99eceb49fc734e848266733167cf76880c7d6fd4726aab74aad2b0f2bea26e`
   - Mark toy. The arXiv source states the three-stage R4T method, and a tiny
     synthetic reward/fan-out exercise can demonstrate the shape of set-level
     rewards and supervision compilation. No released FOLM/diffusion training
     code is available.
2. `8c7e030e2cfe846b58cdaa4d653991565c505d5bf528472df3a6187de5565c51`
   - Mark inconclusive. `task_1_result.tex` contains paper-reported Table 1
     values, but no raw OAR datasets, model outputs, training code, or
     evaluation script are released for recomputation.
3. `8a8d053c547b44507a31b61a1ae46f3fa03fff7478c0a13d9032d2663cdc9f41`
   - Mark inconclusive. `task_2_result.tex` contains paper-reported Table 2
     values, but no Polyvore preprocessing, broad-query generation outputs, or
     model outputs are released for recomputation.
4. `56b0ccdfffbd9e358d372fd11dbd436c1328e26aadc6840efe55c4d9218dea83`
   - Mark toy. The source includes reward-collapse figures and explanatory
     text; a synthetic reward sanity check can show diversity/alignment terms
     penalize collapse, but no raw RL logs are released.
5. `295f630a340b777a2161c8effa7c1efc13fd408e8e435ce266b10b367d3b9376`
   - Mark inconclusive. `query_fanout_efficiency.pdf` and manuscript text
     report latency values, but no executable diffusion retriever or latency
     logs are released.

## Validation

Run the required commands from a clean validation checkout:

1. `uv run --project submissions/efficient-property-aligned-fan-out-retrieval-via-rl-amortized-diffusion python submissions/efficient-property-aligned-fan-out-retrieval-via-rl-amortized-diffusion/generate_evidence.py`
2. `uv run --project submissions/efficient-property-aligned-fan-out-retrieval-via-rl-amortized-diffusion python -m pytest submissions/efficient-property-aligned-fan-out-retrieval-via-rl-amortized-diffusion/tests -q`
3. `uv run pytest -q`
4. `uv run skills/icml-repro-loop/scripts/quick_validate.py skills/icml-repro-loop`
5. `uv run pre-commit run -a`

No GPU or paid API use is planned.
