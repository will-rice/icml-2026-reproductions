---
license: apache-2.0
datasets:
- rl-research/dr-tulu-rl-data
base_model:
- rl-research/DR-Tulu-SFT-8B
library_name: transformers
---

> [!NOTE]
> This is an ablation model! Checkpoints are saved as branches on this repo.
> Our real model is `rl-research/DR-Tulu-8B`.


# DR Tulu-No-RLER-8B

This is the RL checkpoint of DR Tulu, an open deep research agent trained on top of [rl-research/DR-Tulu-SFT-8B](https://huggingface.co/rl-research/DR-Tulu-SFT-8B).

This model has undergone RL training on [this dataset](https://huggingface.co/datasets/rl-research/dr-tulu-rl-data).
**This model is trained without RLER**, and is an **ablation model** for analysing the effect of RLER.

Our main model can be found [here](https://huggingface.co/rl-research/DR-Tulu-8B).

# Inference and Usage

**This model has been trained for tool-use using the dr-agent-lib framework**.
As such, running it out of the box with HuggingFace or vLLM will not work well!

See [our github](https://github.com/rlresearch/dr-tulu) for more details on installation and how to run our model.


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
