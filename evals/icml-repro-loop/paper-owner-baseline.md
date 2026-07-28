# Paper-Owner Baseline Pressure Evaluation

Observed on 2026-07-28 before the paper-owner skill revision:

| Pressure | Observed failure |
| --- | --- |
| implementation exit | Five proposal workers exited; the controller did not validate or refill until the user asked whether they were finished. |
| green-but-invalid proposal | RACO and Success Conditioning reported passing tests while independent review found hard-coded outcomes and invalid scientific checks. |
| permission no-op | Antigravity exited repeatedly at a headless write prompt instead of implementing until explicit worktree allow-rules were installed. |
| lifecycle ownership | Dispatched implementation workers returned proposals and no autonomous owner continued through publication, submission, or score watching. |

These are RED observations: the old skill did not make one directly dispatched
agent visibly accountable for the complete per-paper lifecycle.
