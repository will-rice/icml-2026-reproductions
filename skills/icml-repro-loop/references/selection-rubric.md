# Candidate Selection Rubric

Use current challenge state and inspect source artifacts before scoring. Never label paper-reported values as reproduced; only code-computed outputs can support reproduction. Do not score promises, screenshots, README tables, or paper prose as independent evidence.

## Eligibility

A candidate is eligible only when all of these are true:

- It is not actively claimed, queued, judging, judged, or present in reproduction history. Judged and historical papers remain excluded from candidate selection.
- At least two distinct paper claims are independently testable. A claim is independently testable only when released artifacts or a feasible computation can verify it without treating the paper's reported value as evidence.
- It does not require GPU training. Explicitly requested GPU projects are outside this skill.
- The estimated cumulative paid-API cost for the paper is at most USD 10. More than USD 10 is ineligible, not merely a score penalty.
- Its execution path is not known to be unsafe. Unresolved safety ambiguity pauses selection; a workload proven safe inside an approved isolation boundary may remain eligible.
- Its paper identity, candidate status, artifact availability, and expected execution path have been checked from live or primary sources.

## Base Score

Score each dimension from 0 through 5, then sum them for a maximum base score of 25.

### Direct artifacts

| Score | Evidence available |
| --- | --- |
| 0 | No relevant artifact. |
| 1 | Paper prose, screenshots, or tables only. |
| 2 | Partial code or processed examples, but no direct claim output. |
| 3 | Public data, predictions, checkpoints, or executable code directly supports at least one claim. |
| 4 | Versioned artifacts directly support at least two claims with little reconstruction. |
| 5 | Complete, revision-pinnable raw outputs and executable evaluation artifacts support the target claims. |

### Independently testable claim count

| Score | Distinct claims with an independent test |
| --- | --- |
| 0 | None. |
| 1 | One. |
| 2 | Two. |
| 3 | Three. |
| 4 | Four. |
| 5 | Five or more. |

Count claims, not metrics or repeated dataset slices. Each counted claim needs a separate expected observation and a reproducible test.

### CPU feasibility

| Score | Expected local path |
| --- | --- |
| 0 | No credible CPU path. |
| 1 | More than 24 hours, excessive memory, or major feasibility uncertainty. |
| 2 | 8-24 CPU hours or a large, fragile environment. |
| 3 | 2-8 CPU hours with manageable downloads and dependencies. |
| 4 | At most 2 CPU hours with ordinary workstation resources. |
| 5 | Static validation or deterministic evaluation in at most 30 minutes. |

Estimate the complete evidence run, not only a smoke test.

### Provenance

| Score | Traceability |
| --- | --- |
| 0 | Origin cannot be established. |
| 1 | Unattributed copy or indirect mirror only. |
| 2 | Upstream URL exists, but version or lineage is unclear. |
| 3 | Official repository, release, or dataset is identified. |
| 4 | Exact revisions and source URLs can be pinned for core artifacts. |
| 5 | Exact revisions, hashes, acquisition commands, and lineage can be recorded for every input. |

### Licensing

| Score | Permission clarity |
| --- | --- |
| 0 | Known incompatible terms prohibit the intended use. |
| 1 | No license or materially unclear terms. |
| 2 | Some core artifacts are licensed, but important gaps remain. |
| 3 | Core artifacts have explicit compatible terms; only unused or peripheral items are unclear. |
| 4 | Every required artifact has explicit compatible use terms. |
| 5 | Use and redistribution or deployment terms are explicit, compatible, and recorded for every required artifact. |

## Penalties

Apply every relevant penalty after the base score; penalties are cumulative:

| Condition | Penalty |
| --- | ---: |
| Required artifacts are dead or private | -10 |
| Available evidence is self-report only | -5 |
| A required artifact's license is unclear | -2 |

Do not hide an eligibility failure with a high score. Record each penalty and the source observation that caused it.

## Selection Decision

Rank by final score. Before selection, compare the top three eligible candidates side by side. If fewer than three are eligible, document the exhausted pool and compare every eligible candidate.

| Candidate | Base | Penalties | Final | Testable claims | CPU estimate | API estimate | Main risk |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| A |  |  |  |  |  |  |  |
| B |  |  |  |  |  |  |  |
| C |  |  |  |  |  |  |  |

Select the highest-scoring candidate unless a documented tie-break favors stronger direct evidence, lower execution risk, or lower cost. While the state is idle, persist every ineligible candidate and reason before continuing with:

```bash
uv run python skills/icml-repro-loop/scripts/state.py reject state/repro-loop.json CANDIDATE_JSON
```

`reject` preserves the idle phase; do not use a phase transition. Stop only after
selecting an eligible candidate or documenting an exhausted eligible pool.
