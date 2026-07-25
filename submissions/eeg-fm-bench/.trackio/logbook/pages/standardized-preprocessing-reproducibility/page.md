# Standardized preprocessing reproducibility


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_38ac6f5dc2b0", "created_at": "2026-07-25T15:19:24+00:00", "title": "standardized-preprocessing-reproducibility"}
-->
## Verdict: verified locally

**Computed evidence:** exact method bodies from the pinned release are AST-extracted, hashed, and executed on seeded synthetic EEG for ADFTD and Workload configurations. Channel-name standardization, 256 Hz resampling, and two 10-second windows are repeat-identical. This does not substitute for unavailable raw-dataset validation.
