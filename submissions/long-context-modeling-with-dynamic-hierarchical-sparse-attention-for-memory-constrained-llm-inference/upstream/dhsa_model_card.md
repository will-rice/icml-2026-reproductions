---
license: mit
language:
- en
---

# Dynamic Hierarchical Sparse Attention (DHSA)

This repository hosts the boundary predictor weights used in *[Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](https://arxiv.org/pdf/2510.24606)* (NeurIPS 2025 Workshop on Efficient Reasoning).

## Overview

The boundary predictor is a lightweight transformer module trained to dynamically segment long text sequences into variable-length chunks based on semantic boundaries.
It forms the first stage of Dynamic Hierarchical Sparse Attention (DHSA), enabling large language models to efficiently process long contexts by predicting where to attend sparsely.

## Model Architecture
The boundary predictor consists of three main parts:
1. Shared Encoder – Uses attention and pooling layers to capture the left and right context around each token.
2. Feature Fusion – Combines the two contextual features along with their difference, product, and similarity to represent local semantic changes.
3. MLP Classifier – Takes the fused features and predicts whether a given position marks a boundary.

This lightweight design efficiently identifies semantic shifts in long text sequences.

## Training Data

The predictor was trained on diverse long-context datasets combining multiple reasoning and QA sources:

* [Long Data Collections](https://huggingface.co/datasets/togethercomputer/Long-Data-Collections)
* [Trivia QA](https://huggingface.co/datasets/mandarjoshi/trivia_qa)
* [ChatQA2-Long-SFT](https://huggingface.co/datasets/nvidia/ChatQA2-Long-SFT-data)

Data were automatically annotated using an internal semantic-boundary labeling process based on similarity shifts between consecutive key embeddings.

## Citation

If you use this model, please cite:

```
@inproceedings{xiong2025long,
  title={Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs},
  author={Xiong, Siheng and Zou, Joe and Fekri, Faramarz and Cho, Yae Jee},
  booktitle={NeurIPS 2025 Workshop on Efficient Reasoning}
}
```
