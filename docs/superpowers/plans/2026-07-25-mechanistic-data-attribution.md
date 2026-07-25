# Implementation Plan - Mechanistic Data Attribution Reproduction

## Goal
Build independently executable evidence for paper `PQaxfoEcRc` ("Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units"), execute strict TDD, deploy to a dedicated HuggingFace Space, verify exact SHA, submit, monitor bounded judging, execute 1 improvement round, and complete.

## Steps

1. **Setup Project Directory & Dependencies**
   - Create `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/`
   - Setup `pyproject.toml`, module layout `src/mechanistic_data_attribution_repro/`, `tests/`, `README.md`.
   - Setup `uv` virtual environment and verify pytest.

2. **Strict TDD Implementation**
   - Task 1: Write failing tests for model probes and mechanistic data attribution calculation (`test_probe.py`, `test_attribution.py`). Implement `probe.py` and `attribution.py`.
   - Task 2: Write failing tests for structural pattern analysis (`test_patterns.py`). Implement `patterns.py`.
   - Task 3: Write failing tests for causal intervention suite (`test_intervention.py`). Implement `intervention.py`.
   - Task 4: Write failing tests for CLI and evidence bundle generation (`test_cli.py`). Implement `cli.py` to generate `results.json`, `measurements.csv`, `provenance.json`, and `repro-bundle.tar.gz`.

3. **Validation & Polish**
   - Run submission pytest suite.
   - Build Trackio logbook and HTML poster (`poster.html`, `poster_embed.html`, `poster_preview.png`).
   - Run `uv run pre-commit run -a` and ensure 0 errors.

4. **Space Deployment & Verification**
   - Create and deploy dedicated HuggingFace Space `wrice/repro-mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units`.
   - Verify deployed exact git SHA.

5. **Submission, Bounded Judging & Improvement**
   - Refresh challenge state and submit Space URL.
   - Bounded judging monitoring round.
   - Execute 1 improvement iteration if needed.
   - Complete attempt.
