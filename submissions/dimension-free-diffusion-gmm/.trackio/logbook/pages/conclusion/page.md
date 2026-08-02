# Conclusion


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_89c586939c1c", "created_at": "2026-07-22T14:04:32+00:00", "title": "Reproduction bundle", "artifact": "repro-dimension-free-convergence-of-diffusion-models-for-approximate-gaussian-mixtures/repro-bundle:v2", "artifact_type": "dataset"}
-->
**📦 Artifact** `repro-dimension-free-convergence-of-diffusion-models-for-approximate-gaussian-mixtures/repro-bundle:v2` · dataset

https://huggingface.co/buckets/wrice/repro-dimension-free-convergence-of-diffusion-models-for-approximate-gaussian-mixtures-artifacts#repro-dimension-free-convergence-of-diffusion-models-for-approximate-gaussian-mixtures/repro-bundle:v2


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_5a7e0e3d2b4d", "created_at": "2026-07-22T14:06:47+00:00", "title": "Download and rerun"}
-->
The reproduction bundle contains the NumPy implementation, tests, deterministic JSON/CSV evidence, pinned provenance, poster source, and gate report. After publication, download the `repro-bundle:v2` artifact, extract it, run `uv sync --frozen`, then run `uv run pytest -q` and `uv run python -m diffusion_gmm_repro.cli --output-dir evidence`. The outputs are numerical toy audits and must not be interpreted as replacements for the paper proofs.
