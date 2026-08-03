# VenusBench-Mobile Reproduction Evidence

This reproduction verifies released-artifact claims for VenusBench-Mobile from the pinned `inclusionAI/UI-Venus` `VenusBench-Mobile` branch at commit `5b2c618ef146ea38890ea35dca8b07ec2d0284dd`.

Verified evidence:

- The released README declares 27 Android apps, 149 primary tasks, and 80 stability variants.
- The released metadata contains 189 task records in both `task_instance_goal.json` and `android_world/task_metadata.json`, plus 116 AndroidWorld baseline records.
- Every VenusBench-Mobile metadata record carries PUDAM ability keys `p`, `u`, `d`, `a`, and `m`; `utils/pudam_stats.py` maps them to Perception, Understanding, Decision, Action, and Memory.
- The released artifacts expose hybrid verification: metadata uses both programmatic (`p`) and MLLM (`m`) evaluation methods, `suite_utils.py` calls task success checks, and `android_world/policy/verification.py` implements an LLM-backed verifier.

Partial evidence:

- The stability protocol is present, but the released README and scripts name the five modes as Original, Question Variation, Chinese, Mobile Dark mode, and Pad mode. The exact challenge wording about a distinct min/max setting-variant mode was not found in the released scripts, and this reproduction does not rerun Android emulator episodes to recompute Table 4 agent pass rates.
