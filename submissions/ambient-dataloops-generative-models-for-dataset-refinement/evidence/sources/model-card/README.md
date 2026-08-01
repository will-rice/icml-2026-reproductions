---
license: cc-by-nc-4.0
library_name: diffusers
pipeline_tag: text-to-image
tags:
- diffusion
- text-to-image
- ambient diffusion
- low-quality data
- synthetic data
---

# Ambient Dataloops: Generative Models for Dataset Refinement

## Model Description

Ambient Dataloops is an iterative framework for refining datasets that makes it easier for diffusion models to learn the underlying data distribution. It not only uses uses low-quality, synthetic, and out-of-distribution images to improve the quality of diffusion models, but in turn uses the model to improve the quality of those samples. Just like the other Ambient family approaches, Ambient Dataloops extracts valuable signal from all available images during training, including data typically discarded as "low-quality, unlike traditional approaches that rely on highly curated datasets.

This model card is for a text-to-image diffusion model trained on 8-H100 GPUs only. The key innovation over [Ambient Omni](https://huggingface.co/giannisdaras/ambient-o) is the refinement of low-quality synthetic data, previously only used as "noisy" samples.

## Architecture

Ambient-o builds upon the [MicroDiffusion](https://github.com/SonyResearch/micro_diffusion) cobebase -- we use a Mixture of Experts Diffusion Transformer totaling ~1.1B parameters.

## Text-to-Image Results


Ambient Dataloops demonstrates improvements in text-to-image generation, compared to the baseline of Ambient Omni which does not refine its low-quality data.


### Training Data Composition

The model was trained on a diverse mixture of datasets:
- **Conceptual Captions (CC12M)**: 12M image-caption pairs
- **Segment Anything (SA1B)**: 11.1M high-resolution images with LLaVA-generated captions
- **JourneyDB**: 4.4M synthetic image-caption pairs from Midjourney
- **DiffusionDB**: 10.7M quality-filtered synthetic image-caption pairs from Stable Diffusion

Data from DiffusionDB were treated as noisy samples, and refined once to obtain the final training set.


### Technical Approach

#### Use synthetic samples
As a first step, use the Ambient Omni algorithm to train an initial diffusion model, treating the samples from DiffusionDB as noisy i.e. only using them for $\sigma >= 2.0$

### Refine synthetic samples
Next, we use the trained model to *refine* the synthetic samples, by using posterior samples. These new synthetic samples are better than before, but not as good as real samples. Thus, we still treat them as noisy, but less so than before i.e. only using them for $\sigma >= 1.0$

## Usage

```python
import torch
from micro_diffusion.models.model import create_latent_diffusion
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# Init model
params = {
    'latent_res': 64,
    'in_channels': 4,
    'pos_interp_scale': 2.0,
}
model = create_latent_diffusion(**params).to('cuda')

# Download weights from HF
model_dict_path = hf_hub_download(repo_id="adrianrm/ambient-dataloops", filename="model.safetensors")
model_dict = {}
with safe_open(model_dict_path, framework="pt", device="cpu") as f:
   for key in f.keys():
       model_dict[key] = f.get_tensor(key)

# Convert parameters to float32 + load
float_model_params = {
    k: v.to(torch.float32) for k, v in model_dict.items()
}
model.dit.load_state_dict(float_model_params)

# Eval mode
model = model.eval()

# Generate images
prompts = [
    "A giraffe standing in an open field next to some rocks.",
    "A bike parked next to a red door on the front of a house.",
    "An apple tree filled with lots of apples.",
    "An empty train station has very nice clocks.",
    "A parking lot filled with buses parked next to each other."
    "Panda mad scientist mixing sparkling chemicals, artstation",
    "the sailor galaxia. beautiful, realistic painting by mucha and kuvshinov and bilibin. watercolor, thick lining, manga, soviet realism",
]
images = model.generate(prompt=prompts, num_inference_steps=30, guidance_scale=5.0, seed=42)
```

## Citation

```bibtex
@article{rodriguez2025ambient,
  title = {Ambient Dataloops: Generative Models for Dataset Refinement},
  author = {Rodriguez-Munoz, A. and Daspit, W. and Klivans, A. and Torralba, A. and Daskalakis, C. and Daras, G.},
  year = {2025},
}
```

## License

The model follows the [license](https://github.com/SonyResearch/micro_diffusion/blob/main/LICENSE) of the MicroDiffusion repo.
