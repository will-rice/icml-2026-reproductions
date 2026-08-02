# Claims and Evidence

This page presents the quantitative audit findings and evidence breakdown for the 6 target claims evaluated in the **MedMosaic** (ICML 2026) reproduction.

## Claim Summary and Status

| # | Claim Description | Status | Primary Evidence / Metric |
|---|-------------------|--------|---------------------------|
| 1 | Benchmark dataset size of 46,701 QA pairs | Falsified | Pinned parquet dataset `data/test.parquet` contains 4,661 rows (10x smaller) |
| 2 | QA category coverage across sound, speech, multi-turn, open-ended | Verified | 7 distinct audio categories verified across 4,661 dataset rows |
| 3 | 13-system benchmark performance (Gemini-2.5-Pro at 68.1%) | Inconclusive | Evaluation scripts/logs omitted from published dataset repository |
| 4 | Audio-removal performance degradation ablation | Inconclusive | Audio paths and questions present; ablation outputs omitted |
| 5 | Reasoning difficulty strata (easy, medium, hard) | Inconclusive | Difficulty labels verified (930 easy, 1396 med, 2329 hard); no accuracy outputs |
| 6 | 72.4% clinical expert review acceptance rate | Inconclusive | Review rubrics and scoring sheets omitted from artifact release |

---

## Detailed Evidence Breakdown

### Claim 1: Dataset Volume Audit (Falsified)

- **Target Claim**: MedMosaic contains 46,701 medical audio question-answer pairs spanning physiological sounds, clinical conversations, and combined speech-sound scenarios (Figure 1).
- **Audit Findings**:
  - Pinned repository: `icml-anon-submission/medmosaic-dataset@a6ea67bd4a65b87248c6651e559656b2c31fa669`
  - Parquet dataset file: `data/test.parquet`
  - **Actual Row Count**: 4,661 rows
  - **Claimed QA Pair Count**: 46,701
  - **Discrepancy Ratio**: 0.099786 (10.02x smaller than claimed)
  - **Rows with Valid Question Text**: 4,655
  - **Rows with Ground-Truth Answer**: 4,655
  - **Rows with Audio File Paths**: 4,661

### Claim 2: QA Category Coverage (Verified)

- **Target Claim**: The benchmark includes multiple QA categories, including sound-only, speech-plus-sound, short and long clinical conversations, multi-turn MCQ, and open-ended QA (Table 1).
- **Category Distribution**:

| QA Category | Parquet Row Count | Audio Folder | Percentage of Dataset |
|-------------|-------------------|--------------|-----------------------|
| Sound Only  | 1,417             | `sound_only` | 30.40%                |
| Speech Only | 1,174             | `speech_only`| 25.19%                |
| Speech+Sound| 1,087             | `speech_sound`| 23.32%               |
| Open Ended  | 691               | `open_ended` | 14.83%                |
| Voice QA    | 180               | `voice_qa_v2`| 3.86%                 |
| Long Form   | 106               | `long_form`  | 2.27%                 |
| Multi-Turn  | 6                 | `multi_turn` | 0.13%                 |
| **Total**   | **4,661**         | **All**      | **100.00%**           |

- **MCQ Structure**: 3,964 rows contain standard 10-option multiple-choice questions. 6 multi-turn rows contain 18 total conversation turns (all 18 marked with ground-truth answer keys).

### Claim 3: 13-System Benchmark Performance

- **Target Claim**: Benchmarking 13 audio-language systems shows Gemini-2.5-Pro is the strongest evaluated model but reaches only about 68.1% weighted accuracy (Table 1).
- **Audit Findings**:
  - The dataset repository includes audio files and parquet metadata, but does not provide raw prediction logs, confusion matrices, or evaluation scripts for Gemini-2.5-Pro or the 12 other baselines.

### Claim 4: Audio-Removal Ablation

- **Target Claim**: Removing audio materially reduces model performance, indicating MedMosaic is not trivially solvable from question text alone (Table 5).
- **Audit Findings**:
  - Question text and options are verified in `data/test.parquet`. Text-only baseline outputs were not included in the dataset release.

### Claim 5: Difficulty Strata Breakdown

- **Target Claim**: Model accuracy generally declines from easy to hard difficulty strata across categories, supporting the benchmark's reasoning-difficulty labels (Table 6).
- **Difficulty Stratification Audit**:

| Difficulty Stratum | Row Count | Percentage |
|--------------------|-----------|------------|
| Hard               | 2,329     | 49.97%     |
| Medium             | 1,396     | 29.95%     |
| Easy               | 930       | 19.95%     |
| Null / Unlabeled   | 6         | 0.13%      |
| **Total**          | **4,661** | **100.00%**|

### Claim 6: Clinical Expert Review

- **Target Claim**: Clinical expert review accepted 72.4% of assessed synthetic QA examples without modification, supporting the synthetic generation pipeline's clinical validity (Section 4).
- **Audit Findings**:
  - The published repository does not include expert review forms, anonymized physician rating tables, or raw validation metrics.
