"""DVPD Model Implementation and Reproduction Evidence Verification."""

import math
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FANCEncoder(nn.Module):
    """Frequency-Adaptive Non-uniform Compression (FANC) Encoder.

    Preserves low-frequency harmonics while pruning high-frequency redundancy
    using non-uniform resolution allocation across spectrogram frequency bins.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 64, num_freq_bins: int = 257):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_freq_bins = num_freq_bins

        # Low-frequency branch (dense resolution, indices 0 to 64)
        self.low_freq_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=(3, 3), padding=(1, 1))

        # High-frequency branch (compressed resolution, non-uniform stride/compression)
        self.high_freq_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))

        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, F, T) where F is frequency bins
        low_f = x[:, :, :64, :]
        high_f = x[:, :, 64:, :]

        out_low = self.low_freq_conv(low_f)  # Shape: (B, embed_dim, 64, T)
        out_high = self.high_freq_conv(high_f) # Shape: (B, embed_dim, compressed_F, T)

        # Upsample high frequency representation back to match frequency height (257-64)
        out_high_res = F.interpolate(out_high, size=(x.shape[2] - 64, x.shape[3]), mode='bilinear', align_corners=False)

        cat_feat = torch.cat([out_low, out_high_res], dim=2)
        out = self.out_proj(cat_feat)
        return out


class FrequencyAwareInteraction(nn.Module):
    """Frequency-Aware Interaction Module.

    Enables cross-branch feature exchange between acoustic predictive branch
    and visual texture diffusion branch.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.query_proj = nn.Conv2d(channels, channels, 1)
        self.key_proj = nn.Conv2d(channels, channels, 1)
        self.val_proj = nn.Conv2d(channels, channels, 1)
        self.scale = 1.0 / math.sqrt(channels)

    def forward(self, acoustic_feat: torch.Tensor, visual_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query_proj(acoustic_feat)
        k = self.key_proj(visual_feat)
        v = self.val_proj(visual_feat)

        attn = torch.softmax(torch.sum(q * k, dim=1, keepdim=True) * self.scale, dim=-1)
        interacted_acoustic = acoustic_feat + attn * v
        interacted_visual = visual_feat + attn * q
        return interacted_acoustic, interacted_visual


class LISAModule(nn.Module):
    """LISA: Layer-wise Interaction and Spectrogram Adaptation Module."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.adapt_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapt_conv(x)


class TLBStrategy(nn.Module):
    """Time-aware Layer Budgeting (TLB) Strategy.

    Dynamically allocates computational budget across diffusion timesteps.
    """

    def __init__(self, num_timesteps: int = 1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.time_embed = nn.Embedding(num_timesteps, 64)

    def get_layer_budget(self, t: torch.Tensor) -> torch.Tensor:
        # Returns budget ratio for given timestep t
        emb = self.time_embed(t)
        ratio = torch.sigmoid(emb.mean(dim=-1))
        return ratio


class DVPDModel(nn.Module):
    """Dual-View Predictive Diffusion (DVPD) Model Architecture.

    Integrates:
    - Predictive branch (acoustic frequency-domain structural modeling)
    - Diffusion branch (visual texture refinement)
    - FANC Encoder
    - Frequency-Aware Interaction
    - LISA & TLB modules
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 64):
        super().__init__()
        self.fanc_encoder = FANCEncoder(in_channels=in_channels, embed_dim=embed_dim)

        # Predictive Branch
        self.predictive_branch = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            LISAModule(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        )

        # Diffusion Branch
        self.diffusion_branch = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            LISAModule(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        )

        self.interaction = FrequencyAwareInteraction(embed_dim)
        self.tlb = TLBStrategy()
        self.head = nn.Conv2d(embed_dim, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        feat = self.fanc_encoder(x)

        pred_feat = self.predictive_branch(feat)
        diff_feat = self.diffusion_branch(feat)

        interacted_pred, interacted_diff = self.interaction(pred_feat, diff_feat)
        combined = interacted_pred + interacted_diff
        out = self.head(combined)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_conv_macs(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Measure multiply-accumulate operations of every Conv2d via forward hooks.

    MACs per conv = Cin/groups * Cout * Kh * Kw * Hout * Wout, using the
    actual output shapes observed during a real forward pass.
    """
    macs = 0

    def hook(module: nn.Conv2d, args, output) -> None:
        nonlocal macs
        out_h, out_w = output.shape[-2], output.shape[-1]
        kernel_h, kernel_w = module.kernel_size
        macs += (
            (module.in_channels // module.groups)
            * module.out_channels
            * kernel_h
            * kernel_w
            * out_h
            * out_w
        )

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model(torch.zeros(input_shape))
    for handle in handles:
        handle.remove()
    return macs


def experiment_architecture_and_macs() -> Dict[str, Any]:
    """Real forward pass, measured parameters, and measured conv MACs.

    These numbers describe THIS small-scale implementation only; they are not
    comparable to the paper's full-scale DVPD or PGUSE models and must not be
    used to claim the paper's 35%/40% efficiency ratios.
    """
    torch.manual_seed(3)
    model = DVPDModel(in_channels=1, embed_dim=64)
    model.eval()
    shape = (1, 1, 257, 100)
    with torch.no_grad():
        out = model(torch.randn(*shape), torch.tensor([10]))
    return {
        "forward_output_matches_input_shape": bool(out.shape == torch.Size(shape)),
        "toy_model_parameters": count_parameters(model),
        "toy_model_conv_macs_per_forward": count_conv_macs(model, shape),
        "comparable_to_paper_models": False,
    }


def experiment_interaction_coupling() -> Dict[str, Any]:
    """Measure that the interaction module actually exchanges information.

    Perturbing the visual-branch input must change the acoustic-branch output
    through the cross-branch attention, and vice versa; the measured response
    norms quantify the coupling of the dual-view design.
    """
    torch.manual_seed(5)
    interaction = FrequencyAwareInteraction(channels=8)
    interaction.eval()
    acoustic = torch.randn(1, 8, 16, 16)
    visual = torch.randn(1, 8, 16, 16)
    with torch.no_grad():
        base_acoustic, base_visual = interaction(acoustic, visual)
        pert_acoustic, _ = interaction(acoustic, visual + 0.5)
        _, pert_visual = interaction(acoustic + 0.5, visual)
    return {
        "acoustic_response_to_visual_perturbation": round(
            float(torch.norm(pert_acoustic - base_acoustic).item()), 4
        ),
        "visual_response_to_acoustic_perturbation": round(
            float(torch.norm(pert_visual - base_visual).item()), 4
        ),
    }


def experiment_fanc_band_allocation() -> Dict[str, Any]:
    """Measure FANC's non-uniform resolution and compute allocation per band."""
    encoder = FANCEncoder(in_channels=1, embed_dim=64)
    encoder.eval()
    frames = torch.zeros(1, 1, 257, 100)
    with torch.no_grad():
        low_rows = encoder.low_freq_conv(frames[:, :, :64, :]).shape[2]
        high_rows = encoder.high_freq_conv(frames[:, :, 64:, :]).shape[2]
    low_macs = count_conv_macs(encoder.low_freq_conv, (1, 1, 64, 100))
    high_macs = count_conv_macs(encoder.high_freq_conv, (1, 1, 193, 100))
    return {
        "low_band_input_rows": 64,
        "low_band_representation_rows": int(low_rows),
        "high_band_input_rows": 193,
        "high_band_representation_rows": int(high_rows),
        "high_band_compression_factor": round(193 / high_rows, 2),
        "macs_per_input_row_low_band": round(low_macs / 64),
        "macs_per_input_row_high_band": round(high_macs / 193),
    }


def make_toy_denoising_batch(
    rng: torch.Generator, batch: int = 8, freq: int = 129, time: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthetic harmonic spectrograms plus noise, with clean targets."""
    rows = torch.arange(freq, dtype=torch.float32).view(1, 1, freq, 1)
    fundamentals = torch.randint(4, 12, (batch, 1, 1, 1), generator=rng).float()
    harmonic = ((rows % fundamentals) < 1.0).float().expand(batch, 1, freq, time)
    envelope = torch.linspace(1.0, 0.3, freq).view(1, 1, freq, 1)
    clean = harmonic * envelope
    noisy = clean + 0.3 * torch.randn(batch, 1, freq, time, generator=rng)
    return noisy, clean


class UniformEncoder(nn.Module):
    """Ablation stand-in for FANC: one uniform-resolution conv branch."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=(3, 3), padding=(1, 1))
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.conv(x))


def train_denoiser(variant: str, steps: int = 120, seed: int = 13) -> float:
    """Train one small DVPD variant on toy denoising; return held-out MSE.

    Variants: "full", "no_interaction" (branches never exchange features),
    "no_lisa" (identity in place of LISA), "uniform_encoder" (FANC removed).
    "TLBStrategy" is defined in this package but is not wired into the
    forward pass, so a TLB ablation cannot be trained here.
    """
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(97)
    model = DVPDModel(in_channels=1, embed_dim=16)
    if variant == "uniform_encoder":
        model.fanc_encoder = UniformEncoder(1, 16)
    if variant == "no_lisa":
        model.predictive_branch[1] = nn.Identity()
        model.diffusion_branch[1] = nn.Identity()
    if variant == "no_interaction":
        model.interaction.forward = lambda a, v: (a, v)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        noisy, clean = make_toy_denoising_batch(rng)
        loss = F.mse_loss(model(noisy), clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    eval_rng = torch.Generator().manual_seed(1009)
    noisy, clean = make_toy_denoising_batch(eval_rng, batch=16)
    model.eval()
    with torch.no_grad():
        return round(float(F.mse_loss(model(noisy), clean).item()), 5)


def experiment_toy_ablation() -> Dict[str, Any]:
    """Train the full toy model and three ablated variants; measure MSE."""
    eval_rng = torch.Generator().manual_seed(1009)
    noisy, clean = make_toy_denoising_batch(eval_rng, batch=16)
    noisy_mse = round(float(F.mse_loss(noisy, clean).item()), 5)
    results = {variant: train_denoiser(variant) for variant in (
        "full",
        "no_interaction",
        "no_lisa",
        "uniform_encoder",
    )}
    return {
        "heldout_noisy_input_mse": noisy_mse,
        "heldout_denoised_mse_by_variant": results,
        "tlb_note": "TLBStrategy exists in this package but is not connected to the forward pass; its ablation cannot be trained here",
    }
