# Conclusion


---
<!-- trackio-cell
{"type": "code", "id": "cell_d53b48a550db", "created_at": "2026-07-21T16:00:15+00:00", "title": "Generate the judge-aligned evidence bundle", "command": ["uv", "run", "nape-repro"], "exit_code": 0, "duration_s": 5.949}
-->
````bash
$ uv run nape-repro
````

exit 0 · 5.9s


````output
Large fingerprint (1105 > 1000 threshold), processing may be slow
Large fingerprint (1501 > 1000 threshold), processing may be slow
Wrote reproduction bundle to repro_bundle

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_79b0361bc22b", "created_at": "2026-07-21T16:00:19+00:00", "title": "Run all upstream NAPE tests", "command": ["uv", "run", "pytest", "external/NAPE/tests", "-q"], "exit_code": 0, "duration_s": 0.588}
-->
````bash
$ uv run pytest external/NAPE/tests -q
````

exit 0 · 0.6s


````output
........................................................................ [ 29%]
........................................................................ [ 59%]
........................................................................ [ 88%]
............................                                             [100%]
244 passed in 0.33s

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_3f3c2fbd88e1", "created_at": "2026-07-21T16:00:44+00:00", "title": "Run repository quality gates", "command": ["uv", "run", "pre-commit", "run", "-a"], "exit_code": 0, "duration_s": 21.443}
-->
````bash
$ uv run pre-commit run -a
````

exit 0 · 21.4s


````output
check python ast.........................................................Passed
fix end of files.........................................................Passed
trim trailing whitespace.................................................Passed
check for merge conflicts................................................Passed
fix requirements.txt.................................(no files to check)Skipped
prettier-format..........................................................Passed
ruff format..............................................................Passed
ruff (legacy alias)......................................................Passed
ty.......................................................................Passed
pytest...................................................................Passed

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_a5814c8e08cd", "created_at": "2026-07-21T16:01:15+00:00", "title": "Judge-aligned reproduction bundle (six files)", "artifact": "repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets/repro-bundle-v2:v1", "artifact_type": "dataset"}
-->
**📦 Artifact** `repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets/repro-bundle-v2:v1` · dataset

https://huggingface.co/buckets/wrice/repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets-artifacts#repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets/repro-bundle-v2:v1


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_59e397f4ec1e", "created_at": "2026-07-21T16:06:21+00:00", "title": "Rejudging scope and limitations"}
-->
Current public judged score: **1/12**. This expanded logbook is a **rejudging submission**, not a guaranteed score change.

- **Claim 1 - REPRODUCED:** **52 trajectories**, **11,907 operations**, sequence lengths **35-821**, mean **229**, and median **164**, with executable construction-source evidence and human annotation treated as provenance-only.
- **Claim 2 - REPRODUCED FROM RELEASED OUTPUTS:** **126,940 / 186,574 = 68.04%**, mean **65.99%**, median **66.34%**, and **44/52** trajectories above 50%. The original paid frontier-oracle calls were not rerun.
- **Claim 3 - REPRODUCED:** one deterministic adaptation case per each of the 52 released trajectories, not a full per-action rollout, records **50/52 removal**, **52/52 inverse insertion**, **52/52 target preservation**, and **0/52 residual-patch** cases. One fixed fixture separately records **1/1 residual patch** and **1/1 target preservation**. The small trace is limited to ordering and accept/reject evidence.
- **Claims 4-6 - NOT REPLICATED:** no verdict is offered.

The run used local CPU execution and **zero paid API cost (`$0.00`)**. The attached `repro-bundle-v2` dataset contains all six portable files, including exact observed values, counting definitions, pinned revisions, environment, commands, and explicit unreplicated status.
