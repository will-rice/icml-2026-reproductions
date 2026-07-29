import torch
import numpy as np


class QSchedScheduler:
    """Quantization-Aware Scheduler (Q-Sched) for few-step diffusion models.
    
    Q-Sched optimizes timestep discretization and noise scaling schedules for post-training
    quantized diffusion models without altering model weights.
    """

    def __init__(self, num_timesteps: int = 1000, bit_width_w: int = 4, bit_width_a: int = 8):
        self.num_timesteps = num_timesteps
        self.bit_width_w = bit_width_w
        self.bit_width_a = bit_width_a
        self.betas = np.linspace(0.0001, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)

    def get_quantization_noise_bound(self, x: torch.Tensor, bits: int) -> float:
        """Calculate uniform quantization error bound."""
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        scale = (x.max() - x.min()) / (qmax - qmin + 1e-8)
        quantized = torch.clamp(torch.round(x / scale), qmin, qmax) * scale
        return float(torch.mean((x - quantized) ** 2).item())

    def optimize_few_step_schedule(self, num_inference_steps: int = 4) -> np.ndarray:
        """Optimize timestep placement for N-step inference minimizing quantization noise sensitivity."""
        # Baseline uniform linear spacing
        step_ratio = self.num_timesteps // num_inference_steps
        base_steps = np.arange(0, self.num_timesteps, step_ratio)[:num_inference_steps]
        
        # Q-Sched refinement: shift timesteps toward regions of higher alpha curvature
        curvature = np.abs(np.gradient(np.gradient(self.alphas_cumprod)))
        curvature /= curvature.sum()
        
        optimized_steps = []
        cumulative_curv = np.cumsum(curvature)
        target_quantiles = np.linspace(0, 1, num_inference_steps + 1)[1:]
        
        for q in target_quantiles:
            idx = np.searchsorted(cumulative_curv, q)
            optimized_steps.append(min(idx, self.num_timesteps - 1))
            
        return np.array(optimized_steps)
