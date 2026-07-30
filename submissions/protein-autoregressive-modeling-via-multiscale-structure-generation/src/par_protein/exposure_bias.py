import sys
sys.dont_write_bytecode = True
import numpy as np


class NoisyContextLearning:
    def __init__(self, base_noise_std: float = 0.1):
        self.base_noise_std = base_noise_std

    def inject_context_noise(
        self, context_embeds: np.ndarray, scale_idx: int, seed: int = 42
    ) -> np.ndarray:
        rng = np.random.RandomState(seed + scale_idx)
        scale_std = self.base_noise_std / (1 + scale_idx)
        noise = rng.normal(0.0, scale_std, size=context_embeds.shape)
        return context_embeds + noise


class ScheduledSampling:
    def __init__(
        self,
        strategy: str = "linear",
        start_prob: float = 1.0,
        end_prob: float = 0.1,
        total_steps: int = 1000,
    ):
        self.strategy = strategy
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.total_steps = total_steps

    def get_teacher_forcing_probability(self, current_step: int) -> float:
        if current_step <= 0:
            return self.start_prob
        if current_step >= self.total_steps:
            return self.end_prob

        progress = current_step / self.total_steps

        if self.strategy == "linear":
            prob = self.start_prob - progress * (self.start_prob - self.end_prob)
        elif self.strategy == "exponential":
            decay_rate = np.log(self.start_prob / max(self.end_prob, 1e-6))
            prob = self.start_prob * np.exp(-decay_rate * progress)
        elif self.strategy == "inverse_sigmoid":
            k = 10.0
            prob = k / (k + np.exp(progress * k))
            prob = self.end_prob + (self.start_prob - self.end_prob) * prob
        else:
            raise ValueError(f"Unknown scheduled sampling strategy: {self.strategy}")

        return float(np.clip(prob, self.end_prob, self.start_prob))
