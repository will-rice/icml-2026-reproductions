import sys
sys.dont_write_bytecode = True
import numpy as np
from .multiscale import MultiscaleDownsampler, ScaleRepresentation


class AutoregressiveTransformer:
    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def compute_conditional_embeddings(
        self, prev_scale_rep: ScaleRepresentation, target_length: int
    ) -> np.ndarray:
        num_coarse = prev_scale_rep.coords.shape[0]
        rng = np.random.RandomState(seed=42 + prev_scale_rep.scale_idx)
        proj_weight = rng.randn(3, self.embed_dim) / np.sqrt(3)
        raw_embeds = np.dot(prev_scale_rep.coords, proj_weight)

        repeat_factor = int(np.ceil(target_length / num_coarse))
        upsampled = np.repeat(raw_embeds, repeat_factor, axis=0)[:target_length]

        pos = np.arange(target_length)[:, None]
        dim = np.arange(self.embed_dim)[None, :]
        pe = np.sin(pos / (10000 ** (2 * (dim // 2) / self.embed_dim)))

        return upsampled + pe


class FlowBackboneDecoder:
    def __init__(self, num_steps: int = 10):
        self.num_steps = num_steps

    def decode(
        self, cond_embeds: np.ndarray, init_coords: np.ndarray | None = None
    ) -> np.ndarray:
        target_len = cond_embeds.shape[0]
        if init_coords is None:
            rng = np.random.RandomState(seed=123)
            current_coords = rng.randn(target_len, 3)
        else:
            current_coords = init_coords.copy()

        dt = 1.0 / self.num_steps
        for t_step in range(self.num_steps):
            t = t_step * dt
            velocity = np.sin(cond_embeds[:, :3]) * np.cos(t * np.pi) - 0.1 * current_coords
            current_coords = current_coords + dt * velocity

        return current_coords


class PARModel:
    def __init__(self, scale_factors: tuple[int, ...] = (4, 2, 1), embed_dim: int = 64):
        self.downsampler = MultiscaleDownsampler(scale_factors=scale_factors)
        self.transformer = AutoregressiveTransformer(embed_dim=embed_dim)
        self.decoder = FlowBackboneDecoder(num_steps=10)

    def generate_backbone(self, initial_coarse_coords: np.ndarray) -> dict[str, np.ndarray]:
        scales = self.downsampler.downsample(initial_coarse_coords)
        generated_scales = {}

        current_coords = scales[0].coords
        generated_scales[f"scale_{scales[0].scale_idx}"] = current_coords

        for i in range(1, len(scales)):
            target_scale = scales[i]
            prev_scale_rep = ScaleRepresentation(
                scale_idx=scales[i - 1].scale_idx,
                num_residues=current_coords.shape[0],
                coords=current_coords,
                downsample_factor=scales[i - 1].downsample_factor,
            )
            cond_embeds = self.transformer.compute_conditional_embeddings(
                prev_scale_rep, target_scale.coords.shape[0]
            )
            current_coords = self.decoder.decode(cond_embeds)
            generated_scales[f"scale_{target_scale.scale_idx}"] = current_coords

        return generated_scales
