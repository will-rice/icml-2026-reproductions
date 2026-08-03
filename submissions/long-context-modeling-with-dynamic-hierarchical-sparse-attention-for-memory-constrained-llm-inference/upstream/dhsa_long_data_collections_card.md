---
license: mit
---

# DHSA_Long-Data-Collections

A length-bucketed release of [togethercomputer/Long-Data-Collections](https://huggingface.co/datasets/togethercomputer/Long-Data-Collections), used in *[Long-Context Modeling with Dynamic Hierarchical Sparse Attention for Memory-Constrained LLM Inference](https://arxiv.org/pdf/2510.24606)* (ICML 2026 Spotlight).

Each example is assigned to exactly one length bucket based on token count with `meta-llama/Llama-3.1-8B-Instruct` (`add_special_tokens=True`):

| Bucket | Token range |
|--------|-------------|
| `lt_8k` | [0, 8K) |
| `8k_16k` | [8K, 16K) |
| `16k_32k` | [16K, 32K) |
| `32k_64k` | [32K, 64K) |
| `64k_128k` | [64K, 128K) |
| `gt_128k` | ≥ 128K |

- **Pretrain** examples are bucketed by the `text` field.
- **Fine-tune** examples are bucketed by the `prompt` field.

## Directory layout

```
Long-Data-Collections/
├── README.md
├── length_bucket_summary.json
├── pretrain/
│   ├── lt_8k/
│   │   ├── arxiv_doc_to_abs.jsonl.zst
│   │   ├── NI_decontaminated_materialized.jsonl.zst
│   │   ├── P3_decontaminated_materialized.jsonl.zst
│   │   ├── pile_sub.jsonl.zst
│   │   ├── rp_sub.jsonl.zst
│   │   └── ul2_plus_oscar_en.jsonl.zst
│   ├── 8k_16k/
│   ├── 16k_32k/
│   ├── 32k_64k/
│   ├── 64k_128k/
│   └── gt_128k/
└── fine-tune/
    ├── lt_8k/
    │   ├── booksum.jsonl.zst
    │   └── natural_questions_10_200_docs.jsonl.zst
    ├── 8k_16k/
    ├── 16k_32k/
    ├── 32k_64k/
    ├── 64k_128k/
    └── gt_128k/
```

All files are zstd-compressed JSONL (`.jsonl.zst`). Each line is a JSON object preserving the original fields (`text`, and optionally `meta` / `metadata` for pretrain; `text`, `prompt`, `completion` for fine-tune).

## Example counts by bucket

### Pretrain (bucketed by `text` length)

| Dataset | lt_8k | 8k_16k | 16k_32k | 32k_64k | 64k_128k | gt_128k | **Total** |
|---------|------:|-------:|--------:|--------:|---------:|--------:|----------:|
| arxiv_doc_to_abs | 377,854 | 559,745 | 436,259 | 150,632 | 28,557 | 5,258 | 1,558,305 |
| rp_sub | 913,818 | 11,265 | 3,893 | 1,019 | 368 | 151 | 930,514 |
| ul2_plus_oscar_en | 3,148,215 | 342,739 | 60,306 | 9,213 | 1,804 | 236 | 3,562,513 |
| pile_sub | 1,878,091 | 43,443 | 12,818 | 3,022 | 1,799 | 1,285 | 1,940,458 |
| NI_decontaminated_materialized | 460,066 | 117,188 | 567 | 210 | 12 | 0 | 578,043 |
| P3_decontaminated_materialized | 757,542 | 56,465 | 8 | 0 | 0 | 0 | 814,015 |
| **All pretrain** | **7,535,586** | **1,130,845** | **513,851** | **164,096** | **32,540** | **6,930** | **9,383,848** |

### Fine-tune (bucketed by `prompt` length)

| Dataset | lt_8k | 8k_16k | 16k_32k | 32k_64k | 64k_128k | gt_128k | **Total** |
|---------|------:|-------:|--------:|--------:|---------:|--------:|----------:|
| booksum | 7,909 | 1,207 | 413 | 64 | 5 | 2 | 9,600 |
| natural_questions_10_200_docs | 21,370 | 25,879 | 41,639 | 69 | 0 | 0 | 88,957 |
| **All fine-tune** | **29,279** | **27,086** | **42,052** | **133** | **5** | **2** | **98,557** |

Full counts are also available in `length_bucket_summary.json`.

## Dataset description

This collection compiles long-context datasets for training and evaluating models that require extensive comprehension over large text inputs. It is derived from the public Together AI release and reorganized into length buckets for long-context experiments.

### Pretrain data (`pretrain/`)

| File | Description |
|------|-------------|
| `rp_sub` | RedPajama book subset — diverse literary text |
| `arxiv_doc_to_abs` | RedPajama ArXiv papers with abstract appended after the paper body |
| `ul2_plus_oscar_en` | UL2-style fill-in-the-blank completions (LAION Open-Instruction-Generalist) |
| `pile_sub` | Subsample of The Pile |
| `NI_decontaminated_materialized` | Natural Instructions, decontaminated against HELM core scenarios |
| `P3_decontaminated_materialized` | Public Pool of Prompts (P3), decontaminated against HELM core scenarios |

### Fine-tune data (`fine-tune/`)

| File | Description |
|------|-------------|
| `natural_questions_10_200_docs` | Multi-passage QA from Natural Questions (10–200 Wiki passages per question) |
| `booksum` | Long-context book summarization |

Fine-tune records contain `prompt` (context + instruction), `completion` (target answer/summary), and `text` (`prompt` + `completion`).

## Usage

Load a specific bucket and file with the Hugging Face `datasets` library:

```python
from datasets import load_dataset

ds = load_dataset(
    "json",
    data_files="pretrain/16k_32k/arxiv_doc_to_abs.jsonl.zst",
    split="train",
)
```

Or stream locally with zstd:

```python
import json, zstandard as zstd

path = "pretrain/8k_16k/rp_sub.jsonl.zst"
with open(path, "rb") as f:
    with zstd.ZstdDecompressor().stream_reader(f) as reader:
        for line in reader:
            row = json.loads(line)
            ...
```

## Licensing

Please refer to the original sources of each sub-dataset for their respective licenses. This dataset is a length-bucketed release of [togethercomputer/Long-Data-Collections](https://huggingface.co/datasets/togethercomputer/Long-Data-Collections); upstream licensing terms apply.

## Citation

```bibtex
@inproceedings{xionglong,
  title={Long-Context Modeling with Dynamic Hierarchical Sparse Attention for Memory-Constrained LLM Inference},
  author={Xiong, Siheng and Zou, Joe and Fekri, Faramarz and Cho, Yae Jee},
  booktitle={Forty-third International Conference on Machine Learning}
}
```
