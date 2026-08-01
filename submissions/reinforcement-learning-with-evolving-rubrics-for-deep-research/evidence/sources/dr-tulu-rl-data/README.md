---
dataset_info:
  features:
  - name: source
    dtype: string
  - name: question_type
    dtype: string
  - name: messages
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: ground_truth
    dtype: string
  - name: dataset
    dtype: string
  splits:
  - name: train
    num_bytes: 5950792
    num_examples: 4881
  download_size: 2792178
  dataset_size: 5950792
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: odc-by
---
> [!NOTE]
> For full information, go check out the Dr Tulu paper [here](https://arxiv.org/abs/2511.19399).

<img src="https://huggingface.co/rl-research/DR-Tulu-SFT-8B/resolve/main/dr_tulu_logo.png" alt="Figure 1" width="500"/>

# DR Tulu RL Data

This dataset contains the RL training data for DR Tulu, containing prompts and search-based rubrics generated from OpenScholar and SearchArena prompts, with rubrics generated using GPT-4.1.

**Important**: This does *not* contain the RaR datasets we use in final RL training, but only the OpenScholar and SearchArena subsets. For the RaR data, we use data from:

1. [anisha2102/RaR-Science-20k-o3-mini](https://huggingface.co/datasets/anisha2102/RaR-Science-20k-o3-mini)
2. [anisha2102/RaR-Medicine-20k-o3-mini](https://huggingface.co/datasets/anisha2102/RaR-Medicine-20k-o3-mini)

We sample the first 3000 samples from RaR-Science, and the first 1000 samples from RaR-Medicine.
We will supply code for converting this data into a setup suitable for our training code in our [github](https://allenai.org/papers/drtulu).

## License

This dataset is licensed under ODC-BY. It is intended for research and educational use in accordance with [Ai2's Responsible Use Guidelines](https://allenai.org/responsible-use).
