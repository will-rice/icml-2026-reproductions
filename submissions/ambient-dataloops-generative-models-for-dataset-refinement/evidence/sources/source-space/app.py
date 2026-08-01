import spaces
import gradio as gr
import numpy as np
import random
import torch
import torch
from micro_diffusion.models.model import create_latent_diffusion
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from PIL import Image

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
model = model.eval()


dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

@spaces.GPU()
def infer(prompt, seed=42, randomize_seed=False, guidance_scale=5.0, num_inference_steps=28, progress=gr.Progress(track_tqdm=True)):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    images = model.generate(prompt=[prompt], num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, seed=seed)
    image = images[0]
    image = image.detach().cpu()
    image = image.permute(1, 2, 0)  # [H, W, C]
    image = (image * 255).clamp(0, 255).to(torch.uint8).numpy()
    image = Image.fromarray(image)
    return image, seed


examples = [
    "A giraffe standing in an open field next to some rocks.",
    "A bike parked next to a red door on the front of a house.",
    "An apple tree filled with lots of apples.",
    "An empty train station has very nice clocks.",
    "A parking lot filled with buses parked next to each other."
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

with gr.Blocks(css=css) as demo:

    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# Ambient Dataloops text2image model
[[paper](https://arxiv.org/abs/2601.15417)]
[[blog](https://adrianrm99.github.io/_pages/ambient_dataloops)] [[model](https://huggingface.co/adrianrm/ambient-dataloops)] [[license](https://github.com/adrianrm99/ambient_dataloops/blob/main/text-to-image/LICENSE)]
        """)

        with gr.Row():

            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )

            run_button = gr.Button("Run", scale=0)

        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():

                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1,
                    maximum=15,
                    step=0.1,
                    value=5.0,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=28,
                )

        gr.Examples(
            examples = examples,
            fn = infer,
            inputs = [prompt],
            outputs = [result, seed],
            cache_examples="lazy"
        )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [prompt, seed, randomize_seed, guidance_scale, num_inference_steps],
        outputs = [result, seed]
    )

demo.launch()
