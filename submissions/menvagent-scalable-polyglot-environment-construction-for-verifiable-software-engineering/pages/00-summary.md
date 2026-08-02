# Executive Summary: MEnvAgent Reproduction

This reproduction assesses the empirical claims of **MEnvAgent: Scalable Polyglot Environment Construction for Verifiable Software Engineering** using immutable datasets and released code repositories.

## Key Findings

- **MEnvBench Dataset (Claim 2 - Verified)**: Verified 1,000 task instances across 200 unique repositories in 10 programming languages (C, C++, Go, Java, JavaScript, PHP, Python, Ruby, Rust, TypeScript; 100 tasks per language).
- **MEnvData-SWE Release (Claim 6 - Falsified)**: MEnvData-SWE dataset matches claimed counts of 3,005 task instances across 942 repositories in 10 languages. However, `MEnvData-SWE-Trajectory` contains **3,918** row entries on HuggingFace LFS rather than the claimed **3,872** trajectories.
- **Architectural Terms (Claim 1 - Toy)**: Pinned source code confirms presence of Planning-Execution-Verification structure and `EnvPatchAgent` terms, but core runtime code is unreleased ("organized for public release").
- **Experimental Claims (Claims 3, 4, 5 - Unavailable)**: Proprietary model evaluations (Kimi-K2, Gemini-3-Flash), component ablations, and fine-tuning trajectories require raw execution logs or unreleased model weights.
