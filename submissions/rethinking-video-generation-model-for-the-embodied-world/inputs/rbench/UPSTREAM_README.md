---
license: cc-by-4.0
task_categories:
- image-to-video
- robotics
language:
- en
tags:
- robotics video generation
- benchmark
size_categories:
- 100-1k
pretty_name: RBench
dataset_summary: >
  RBench is a curated evaluation benchmark designed to systematically assess the capabilities of video-generation models in realistic robotic scenarios.
---

<div align="center">
  <img src="assets/logo.png" width="160"/>
</div>

<h1  align="center" style="font-size: 2.0em; text-decoration: underline;">
Rethinking Video Generation Model for the Embodied World
</h1>

<div align="center">

<a href="https://github.com/DAGroup-PKU/ReVidgen">
  <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github"/>
</a>
<a href="https://arxiv.org/abs/2601.15282">
  <img src="https://img.shields.io/badge/Paper-4B0082?style=for-the-badge&logo=arxiv"/>
</a>
<a href="https://dagroup-pku.github.io/ReVidgen.github.io/">
  <img src="https://img.shields.io/badge/Project-2E8B57?style=for-the-badge&logo=googlechrome"/>
</a>
<a href="https://huggingface.co/datasets/DAGroup-PKU/RoVid-X">
  <img src="https://img.shields.io/badge/Dataset-005FED?style=for-the-badge&logo=huggingface"/>
</a>
<a href="https://huggingface.co/datasets/DAGroup-PKU/RBench">
  <img src="https://img.shields.io/badge/Benchmark-FF8C00?style=for-the-badge&logo=semanticweb"/>
</a>
<a href="https://huggingface.co/spaces/DAGroup-PKU/RBench-Leaderboard">
  <img src="https://img.shields.io/badge/Leaderboard-CDA434?style=for-the-badge&logo=googlesheets"/>
</a>
<a href="https://youtu.be/Ea91ErBMBLM">
  <img src="https://img.shields.io/badge/Video-CC0000?style=for-the-badge&logo=youtube"/>
</a>

</div>



# 🔍 Benchmark Overview

The benchmark is constructed from two complementary perspectives: **task categories** and **robot embodiment types**, covering a total of **650 image-text evaluation cases**.

## 🧩 Task-Oriented Evaluation Set (5 Tasks)

The task-oriented split contains **250 image-text pairs**, with **50 samples per task**, spanning five representative robotic task categories:

- **Common Manipulation**: Everyday object manipulation tasks
- **Long-horizon Planning**: Sequential actions requiring multi-step planning
- **Multi-entity Collaboration**: Interactions involving multiple agents
- **Spatial Relationship**: Reasoning about relative positions and spatial constraints
- **Visual Reasoning**: Tasks requiring visual inference beyond direct observation


## 🤖 Embodiment-Oriented Evaluation Set (4 Embodiments)

The embodiment-oriented split contains **400 image-text pairs**, with **100 samples per embodiment**, covering four mainstream robotic embodiment types:

- **Dual-arm Robots**
- **Humanoid Robots**
- **Single-arm Robots**
- **Quadruped Robots**

This split evaluates whether generative models can correctly reflect embodiment-specific physical structures and action affordances.

# 📦 Data Format

Each evaluation sample is stored in JSON format and includes:
- `name`: Unique sample identifier
- `image_path`: Path to the reference image
- `prompt`: Concise task description
- `robotic manipulator` / `manipulated object`: Key semantic entities
- `view`: Camera viewpoint (e.g., first-person)

Images are provided in JPEG format.

# 🛠️ Usage

This benchmark is intended for:
- Image-to-video (I2V) and video generation evaluation
- Vision-language model (VLM / MLLM) benchmarking

# 📜 License

This dataset is released under the **CC BY 4.0 License**.

# 📚 Citation

If you find this dataset useful, please cite our paper:

```bibtex
@article{deng2026rethinking,
  title={Rethinking Video Generation Model for the Embodied World},
  author={Deng, Yufan and Pan, Zilin and Zhang, Hongyu and Li, Xiaojie and Hu, Ruoqing and Ding, Yufei and Zou, Yiming and Zeng, Yan and Zhou, Daquan},
  journal={arXiv preprint arXiv:2601.15282},
  year={2026}
}
