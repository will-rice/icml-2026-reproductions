# Neural Thickets Reproduction Design

Attempt: `228a446e-f3c6-4ee1-8d80-28b6d1226520`
Paper: `92oF5bU4cU`
Snapshot: `cd566b1fc072468cea13824a2382d9be6916bd5ffb684b5affcbfa814f753528`
Title: Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights

## Upstream Pins

- Paper: `arxiv:2603.12228`
- OpenReview: `92oF5bU4cU`
- Official code: `github:sunrainyg/RandOpt@536df0a308f3990b6270c991fbb96bd0b779a58e`
- Project page: `https://thickets.mit.edu/`

## Target Claims

1. Large pretrained models are surrounded by dense neighborhoods of task-specialized perturbations, unlike smaller needle-in-haystack regimes.
2. Solution density and diversity around Qwen2.5 instruction-tuned models increase with model scale.
3. Randomly sampled perturbations exhibit diverse task specialties rather than all acting as generalists.
4. RandOpt samples random parameter perturbations, selects top performers, and ensembles predictions, matching or exceeding PPO, GRPO, ES, and related baselines in many LLM post-training settings.
5. RandOpt accuracy improves with population size and depends on sufficient pretrained model scale.

## Evidence Plan

The reproduction will build a CPU-only audit package around the pinned official repository. It will not rerun GPU-scale Qwen, Llama, OLMo, vLLM, PPO, or GRPO experiments. Instead it will:

- fetch and hash the pinned official files needed for RandOpt, baseline, data, and simple 1D experiment coverage;
- verify the source implements random perturbation sampling, top-k model selection, and majority-vote ensembling;
- verify supported model families and dataset handlers named by the paper are present in the repository;
- run deterministic toy simulations that independently test the qualitative density/diversity and population-size mechanisms without copying paper-reported values;
- mark full scale- and benchmark-metric claims unavailable or toy unless raw released result artifacts are present.

## Expected Limitations

Claims requiring large-model inference or reinforcement-learning baselines will be limited to artifact-backed source verification plus synthetic mechanism checks. Paper-reported tables and figures will not be treated as reproduced measurements.

## Validation Commands

- Generate evidence bundle.
- Run the paper pytest suite.
- Run repository pytest.
- Run `quick_validate.py skills/icml-repro-loop`.
- Run `uv run pre-commit run -a`.
