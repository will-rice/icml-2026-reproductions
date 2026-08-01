---
license: apache-2.0
datasets:
- rl-research/dr-tulu-rl-data
base_model:
- rl-research/DR-Tulu-SFT-8B
library_name: transformers
---

> [!NOTE]
> For full information, go check out the Dr Tulu paper [here](https://arxiv.org/abs/2511.19399).
> We have recently (23/02/2026) updated the model, please check branches for older versions of the model. Our initial release was `step_1000`, our second release was `step_1900`. Our final release is `step_4000` (main and branch).
> Some checkpoint steps are missing, apologies, we simply do not have those checkpoints.

<img src="https://huggingface.co/rl-research/DR-Tulu-SFT-8B/resolve/main/dr_tulu_logo.png" alt="Figure 1" width="500"/>


# DR Tulu-8B

This is the RL checkpoint of DR Tulu, an open deep research agent trained on top of [rl-research/DR-Tulu-SFT-8B](https://huggingface.co/rl-research/DR-Tulu-SFT-8B).

This model has undergone RL training on [this dataset](https://huggingface.co/datasets/rl-research/dr-tulu-rl-data).
For more details on DR Tulu please **read our [paper](https://allenai.org/papers/drtulu)**!

# Inference and Usage

**This model has been trained for tool-use using the dr-agent-lib framework**.
As such, running it out of the box with HuggingFace or vLLM will not work well!

See [our github](https://github.com/rlresearch/dr-tulu) for more details on installation and how to run our model.
Or check out our [demo](https://dr-tulu.github.io/)!

# Evaluation Results

We provide evaluation instructions in [our github](https://github.com/rlresearch/dr-tulu).


| Benchmark | SQAv2 | HealthBench | ResearchQA | DeepResearch Bench | SimpleQA | 2Wiki | WebWalker | Average |
|:----------|:------:|:----------:|:---------:|:-------------------:|:------:|:-------:|-------:|-------:|
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (naive rag) | 40.4 | 16.5 | 56.1 | 33.3 | 52.6 | 18.9 | 8.8 | 32.4 |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (our search pipeline) | 57.2 | 5.9 | 46.3 | 18.2 | 70.5 | 44.0 | 27.9 | 38.6 |
| [DR-Tulu-SFT-8B](https://huggingface.co/rl-research/DR-Tulu-SFT-8B) | 72.3 | 38.1 | 68.5 | 39.0 | 75.5 | 66.5 | 31.9 | 56.0 |
| [DR-Tulu-8B](https://huggingface.co/rl-research/DR-Tulu-8B) (**this model**) | **88.3** | **52.8** |  **75.7** | **45.4** | **75.9** | **68.9** | **39.0** | **63.7** |

For more baselines, explanations of this table, and analysis of results, check out the [Dr Tulu paper](https://allenai.org/papers/drtulu)!

# Intended uses & limitations

This model is licensed under Apache 2.0. It is intended for research and educational use in accordance with [Ai2's Responsible Use Guidelines](https://allenai.org/responsible-use).

## Training

The script used to train this model can be found [here](https://github.com/rlresearch/dr-tulu/blob/rl/rl/open-instruct/train_dr_tulu.sh).

For hyperparameter details, check out the [Dr Tulu paper](http://allenai-web/papers/drtulu).

# Links
- 📝 [DR Tulu Paper](https://allenai.org/papers/drtulu)
- ⚙️ [DR Tulu demo](https://dr-tulu.github.io/)
- 💻 [DR Tulu code](https://github.com/rlresearch/DR-Tulu)
- 🤖 [DR Tulu collection](https://huggingface.co/collections/rl-research/dr-tulu)


# Citation
```
@article{shao2025dr,
  title={DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research},
  author={Shao, Rulin and Asai, Akari and Shen, Shannon Zejiang and Ivison, Hamish and Kishore, Varsha and Zhuo, Jingming and Zhao, Xinran and Park, Molly and Finlayson, Samuel G and Sontag, David and others},
  journal={arXiv preprint arXiv:2511.19399},
  year={2025}
}
```
