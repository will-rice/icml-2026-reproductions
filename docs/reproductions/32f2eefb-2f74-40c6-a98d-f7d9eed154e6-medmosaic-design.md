# MedMosaic Reproduction Design

Attempt: `32f2eefb-2f74-40c6-a98d-f7d9eed154e6`
Paper ID: `OMdQJQwp26`
Owner: `codex-paper-owner-04`
Snapshot: `b09826f921a7d1649e5071df82e7a9f8f6211ac6dfa0e2589aa54651661eb283`
Challenge revision: `81166abbeb76e5f79ff87e51061b5a0306507203`
Challenge file SHA256: `65a632313094067874c7ab2b9f62b87dfb4cf913c7a7052c1c2a29a93ca29940`

## Primary Artifacts

- Paper: `arxiv:2605.00969v2`
- OpenReview: `https://openreview.net/forum?id=OMdQJQwp26`
- Dataset: `icml-anon-submission/medmosaic-dataset@a6ea67bd4a65b87248c6651e559656b2c31fa669`
- Dataset index: `data/test.parquet` from the pinned dataset revision

## Selected Claims

The reproduction targets all six challenge-bound claims:

1. Dataset scale and modality coverage: 46,701 medical audio QA pairs spanning physiological sounds, clinical conversations, and combined speech-sound scenarios.
2. QA category coverage: sound-only, speech-plus-sound, short and long clinical conversations, multi-turn MCQ, and open-ended QA.
3. Benchmark result: Gemini-2.5-Pro is strongest among 13 audio-language systems at about 68.1% weighted accuracy.
4. Audio-removal ablation: removing audio materially reduces model performance.
5. Difficulty labels: model accuracy generally declines from easy to hard.
6. Clinical expert review: 72.4% of assessed synthetic QA examples accepted without modification.

## Evidence Plan

The released dataset index is small enough to audit directly without downloading the 15.7 GB audio payload. Evidence generation will download only `data/test.parquet` from the pinned Hugging Face dataset revision and compute row count, QA type counts, difficulty counts, answer-option shape checks, audio-path folder coverage, and multi-turn row structure.

Claim 1 will be judged against the released artifact. If the pinned dataset index contains 4,661 rows, the bundle will report the paper's 46,701 value as not reproduced from the released artifact rather than copying the paper-reported count.

Claim 2 can be verified structurally if the pinned index contains the expected seven QA categories and folder mappings: `sound_only`, `speech_sound`, `speech_only`, `long_form`, `multi_turn`, `open_ended`, and `voice_qa`.

Claims 3, 4, 5, and 6 require model evaluation logs, audio-removal ablations, per-difficulty model scores, or expert-review adjudication records. Unless those primary result artifacts are present in the pinned dataset repository, the bundle will mark them `inconclusive` and include only structural checks such as difficulty-label counts. Paper tables will not be emitted as reproduced measurements.

## Validation Plan

- Add a small Python package that loads the pinned dataset index with `huggingface_hub`, `pandas`, and `pyarrow`.
- Write tests first for the expected released-row count, category coverage, option format, difficulty coverage, and conservative status assignment.
- Generate `evidence/bundle.json` with pins, commands, row/category counts, claim statuses, and cost `0.00`.
- Include `README.md`, `requirements.txt`, `app.py`, and `pages/report.md` so the Hugging Face Space can be published and scored.
- Validate with the controller's required five commands: evidence generation, paper pytest, root pytest, skill validation, and pre-commit.

## Cost and Safety

The plan is CPU-only and uses public metadata. It does not run clinical model inference, does not download private data, and does not make medical recommendations. Metered cost is expected to remain `0.00` USD.
