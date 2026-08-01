# ThunderAgent Reproduction Design

Attempt: `29efb851-aeef-4989-b74d-a83e2c481384`
Paper: `kR4iOTaAOJ`
Owner: `agy-paper-owner-03`

## Target Claims

1. ThunderAgent abstracts agentic workflows as LLM Programs that unify KV cache, system state, and external tool resources (Section 3).
2. The system adds a program-aware scheduler to improve KV cache hit rates and reduce memory imbalance across agent workflows (Section 4.2).
3. The tool resource manager asynchronously prepares and reuses tool environments, including disk and port resources (Section 4.3).

## Evidence Plan

Use the pinned MIT source repository `ThunderAgent-org/ThunderAgent@7ddc8610270e56d3b109eed8796b3a4360fc67c9`.

The submission will produce CPU-only evidence that:

- pins upstream repository identity, license, and source commit;
- inspects source files in `fixtures/ThunderAgent` for the actual implementation of LLM Programs, system state, and tool resources;
- inspects the program-aware scheduler for KV cache hit rate optimization and memory balance across workflows;
- inspects the tool resource manager for asynchronous preparation and reuse of disk and port resources;
- runs unit tests verifying program state, scheduler queue operations, and tool manager resource pooling;
- generates deterministic JSON evidence detailing the verification status for target claims.

## Limits

The reproduction will not rerun multi-GPU serving benchmarks, LLM inference rollouts, or end-to-end RL throughput evaluation (Figures 5, 6, 9), as these require multi-GPU infrastructure. Benchmark serving speedups and RL rollout claims remain unselected/unverified.

## Validation

Write tests first for the evidence builder and project implementation:

- claim bindings and upstream pins are exact;
- required ThunderAgent source indicators (program abstraction, scheduler, tool manager) are present;
- bundle generation round-trips to JSON;
- status for GPU-heavy claims stays out of the verified target set.

Run:

- `./.venv/bin/python submissions/thunderagent-a-fast-simple-and-program-aware-agentic-inference-system/generate_evidence.py`
- `./.venv/bin/python -m pytest submissions/thunderagent-a-fast-simple-and-program-aware-agentic-inference-system/tests -q`
- `./.venv/bin/python -m pytest -q`
- `./.venv/bin/python skills/icml-repro-loop/scripts/quick_validate.py skills/icml-repro-loop`
