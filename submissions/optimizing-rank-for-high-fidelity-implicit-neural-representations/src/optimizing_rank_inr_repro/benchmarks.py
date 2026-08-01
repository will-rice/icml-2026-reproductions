"""Benchmark suites for validating the 4 claims of Muon INR rank optimization."""

import math
import torch

from .models import SirenINR, VanillaMLP, compute_psnr
from .optimizers import Muon


def generate_synthetic_2d_image(size: int = 32, seed: int = 42) -> torch.Tensor:
    """Generate a deterministic multi-frequency 2D target image [1, size, size]."""
    torch.manual_seed(seed)
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    img = (
        torch.sin(4 * math.pi * grid_x) * torch.cos(4 * math.pi * grid_y)
        + 0.5 * torch.sin(12 * math.pi * grid_x + 8 * math.pi * grid_y)
        + 0.25 * torch.cos(24 * math.pi * grid_x)
    )
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.unsqueeze(0)


def generate_phantom(size: int = 32) -> torch.Tensor:
    """Generate a deterministic Shepp-Logan-style ellipse phantom [1, size, size]."""
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    def ellipse(cx, cy, ax, ay, angle, value):
        ca, sa = math.cos(angle), math.sin(angle)
        xr = (grid_x - cx) * ca + (grid_y - cy) * sa
        yr = -(grid_x - cx) * sa + (grid_y - cy) * ca
        return value * ((xr / ax) ** 2 + (yr / ay) ** 2 <= 1.0).float()

    img = (
        ellipse(0.0, 0.0, 0.85, 0.95, 0.0, 1.0)
        - ellipse(0.0, 0.0, 0.75, 0.85, 0.0, 0.6)
        + ellipse(-0.25, 0.2, 0.18, 0.32, 0.4, 0.35)
        + ellipse(0.25, 0.2, 0.18, 0.32, -0.4, 0.35)
        + ellipse(0.0, -0.35, 0.3, 0.18, 0.0, 0.25)
        + ellipse(0.0, 0.45, 0.08, 0.08, 0.0, 0.5)
    )
    img = img.clamp(0.0, 1.0)
    return img.unsqueeze(0)


def generate_coordinate_grid(size: int = 32) -> torch.Tensor:
    """Generate 2D coordinate grid [-1, 1]^2 of shape [size*size, 2]."""
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)


def build_radon_matrix(size: int = 32, n_angles: int = 8, n_bins: int = 32) -> torch.Tensor:
    """Build a discrete sparse-view Radon operator [n_angles * n_bins, size * size].

    Each pixel center is projected onto the detector axis of every view and
    splatted onto its two nearest detector bins with bilinear weights, so
    A @ image.flatten() gives the sinogram of line integrals.
    """
    coords = generate_coordinate_grid(size=size)
    angles = torch.linspace(0.0, math.pi, n_angles + 1)[:-1]
    rows = []
    half_width = math.sqrt(2.0)
    for theta in angles:
        t = coords[:, 0] * math.cos(theta) + coords[:, 1] * math.sin(theta)
        pos = (t + half_width) / (2.0 * half_width) * (n_bins - 1)
        low = pos.floor().clamp(0, n_bins - 1)
        high = (low + 1).clamp(0, n_bins - 1)
        w_high = pos - low
        w_low = 1.0 - w_high
        view = torch.zeros(n_bins, size * size)
        pixel_idx = torch.arange(size * size)
        view[low.long(), pixel_idx] += w_low
        view[high.long(), pixel_idx] += w_high
        rows.append(view)
    return torch.cat(rows, dim=0) * (2.0 / size)


def make_model_pair(arch: str, seed: int, in_dim: int = 2, out_dim: int = 1):
    """Create identically initialized Adam/Muon model pairs with paired optimizers."""
    torch.manual_seed(seed)
    if arch == "siren":
        model_adam = SirenINR(in_dim=in_dim, hidden_dim=64, out_dim=out_dim, num_layers=4, omega_0=10.0)
        model_muon = SirenINR(in_dim=in_dim, hidden_dim=64, out_dim=out_dim, num_layers=4, omega_0=10.0)
        adam_lr, muon_lr = 1e-3, 0.005
    elif arch == "vanilla_mlp":
        model_adam = VanillaMLP(in_dim=in_dim, hidden_dim=64, out_dim=out_dim, num_layers=4)
        model_muon = VanillaMLP(in_dim=in_dim, hidden_dim=64, out_dim=out_dim, num_layers=4)
        adam_lr, muon_lr = 1e-3, 0.002
    else:
        raise ValueError(f"unknown architecture: {arch}")
    model_muon.load_state_dict(model_adam.state_dict())
    opt_adam = torch.optim.Adam(model_adam.parameters(), lr=adam_lr)
    opt_muon = Muon(model_muon.parameters(), lr=muon_lr)
    return model_adam, model_muon, opt_adam, opt_muon


def train_step(model, optimizer, coords, targets, operator=None):
    """One MSE step; operator maps model output to measurement space when given."""
    optimizer.zero_grad()
    pred = model(coords)
    if operator is not None:
        pred = operator @ pred
    loss = torch.mean((pred - targets) ** 2)
    loss.backward()
    optimizer.step()


def run_claim1_stable_rank_degradation_test(steps: int = 100, seed: int = 42) -> dict:
    """Test Claim 1: Vanilla MLP INR low-frequency bias is accompanied by stable-rank degradation under Adam,

    whereas Muon updates preserve stable rank.
    """
    coords = generate_coordinate_grid(size=32)
    targets = generate_synthetic_2d_image(size=32, seed=seed).reshape(-1, 1)
    mlp_adam, mlp_muon, opt_adam, opt_muon = make_model_pair("vanilla_mlp", seed)

    adam_initial_rank = mlp_adam.average_stable_rank()
    muon_initial_rank = mlp_muon.average_stable_rank()

    for _ in range(steps):
        train_step(mlp_adam, opt_adam, coords, targets)
        train_step(mlp_muon, opt_muon, coords, targets)

    final_adam_rank = mlp_adam.average_stable_rank()
    final_muon_rank = mlp_muon.average_stable_rank()

    adam_rank_drop = adam_initial_rank - final_adam_rank
    muon_rank_drop = muon_initial_rank - final_muon_rank

    preserved_rank = final_muon_rank > final_adam_rank

    return {
        "initial_stable_rank": round(adam_initial_rank, 4),
        "final_adam_stable_rank": round(final_adam_rank, 4),
        "final_muon_stable_rank": round(final_muon_rank, 4),
        "adam_rank_drop": round(adam_rank_drop, 4),
        "muon_rank_drop": round(muon_rank_drop, 4),
        "rank_preserved_by_muon": preserved_rank,
        "status": "verified" if preserved_rank else "unverified",
    }


def run_claim2_image_overfitting_test(steps: int = 100, seed: int = 42) -> dict:
    """Test Claim 2: Rank-regulating Muon updates improve image overfitting quality across multiple INR architectures compared with Adam."""
    coords = generate_coordinate_grid(size=32)
    targets = generate_synthetic_2d_image(size=32, seed=seed).reshape(-1, 1)

    results = {}
    for arch in ("siren", "vanilla_mlp"):
        model_adam, model_muon, opt_adam, opt_muon = make_model_pair(arch, seed)
        for _ in range(steps):
            train_step(model_adam, opt_adam, coords, targets)
            train_step(model_muon, opt_muon, coords, targets)

        with torch.no_grad():
            psnr_adam = compute_psnr(model_adam(coords), targets)
            psnr_muon = compute_psnr(model_muon(coords), targets)

        results[arch] = {
            "psnr_adam": round(psnr_adam, 2),
            "psnr_muon": round(psnr_muon, 2),
            "psnr_gain": round(psnr_muon - psnr_adam, 2),
            "improved": psnr_muon >= psnr_adam,
        }

    all_improved = all(v["improved"] for v in results.values())
    return {
        "architectures": results,
        "all_architectures_improved": all_improved,
        "status": "verified" if all_improved else "unverified",
    }


def run_claim3_sparse_ct_test(steps: int = 200, seed: int = 42, n_angles: int = 8) -> dict:
    """Test Claim 3: Muon improves sparse-view CT reconstruction quality across multiple INR architectures compared with Adam.

    A deterministic ellipse phantom is measured through a discrete sparse-view
    Radon operator (8 views x 32 detector bins -> 256 measurements for 1024
    pixels). Each INR trains only on the sinogram; reconstruction PSNR is
    evaluated against the phantom on the full grid.
    """
    size = 32
    coords = generate_coordinate_grid(size=size)
    phantom = generate_phantom(size=size).reshape(-1, 1)
    radon = build_radon_matrix(size=size, n_angles=n_angles, n_bins=size)
    sinogram = radon @ phantom

    results = {}
    for arch in ("siren", "vanilla_mlp"):
        model_adam, model_muon, opt_adam, opt_muon = make_model_pair(arch, seed)
        for _ in range(steps):
            train_step(model_adam, opt_adam, coords, sinogram, operator=radon)
            train_step(model_muon, opt_muon, coords, sinogram, operator=radon)

        with torch.no_grad():
            recon_psnr_adam = compute_psnr(model_adam(coords), phantom)
            recon_psnr_muon = compute_psnr(model_muon(coords), phantom)

        results[arch] = {
            "recon_psnr_adam": round(recon_psnr_adam, 2),
            "recon_psnr_muon": round(recon_psnr_muon, 2),
            "psnr_gain_db": round(recon_psnr_muon - recon_psnr_adam, 2),
            "improved": recon_psnr_muon >= recon_psnr_adam,
        }

    all_improved = all(v["improved"] for v in results.values())
    return {
        "n_views": n_angles,
        "n_detector_bins": size,
        "n_measurements": n_angles * size,
        "n_pixels": size * size,
        "architectures": results,
        "all_architectures_improved": all_improved,
        "status": "verified" if all_improved else "unverified",
    }


def run_claim4_multidomain_extension_test(steps: int = 100, seed: int = 42) -> dict:
    """Test Claim 4: The reported improvements extend to natural images, medical images, audio, and super-resolution.

    Super-resolution trains on a 16x16 coordinate subgrid and evaluates PSNR on
    the full 32x32 grid; the medical domain fits the CT phantom directly.
    """
    domains = ("natural_image", "medical_phantom", "audio_1d", "super_resolution")
    domain_results = {}

    for domain in domains:
        in_dim = 1 if domain == "audio_1d" else 2
        if domain == "audio_1d":
            t = torch.linspace(-1, 1, 256).unsqueeze(-1)
            signal = (torch.sin(4 * math.pi * t) + 0.5 * torch.sin(12 * math.pi * t)).reshape(-1, 1)
            train_coords, train_targets = t, signal
            eval_coords, eval_targets = t, signal
        elif domain == "medical_phantom":
            train_coords = generate_coordinate_grid(size=32)
            train_targets = generate_phantom(size=32).reshape(-1, 1)
            eval_coords, eval_targets = train_coords, train_targets
        elif domain == "super_resolution":
            full_img = generate_synthetic_2d_image(size=32, seed=seed)
            full_coords = generate_coordinate_grid(size=32)
            low_mask = torch.zeros(32, 32, dtype=torch.bool)
            low_mask[::2, ::2] = True
            low_mask = low_mask.reshape(-1)
            train_coords = full_coords[low_mask]
            train_targets = full_img.reshape(-1, 1)[low_mask]
            eval_coords = full_coords
            eval_targets = full_img.reshape(-1, 1)
        else:  # natural_image
            train_coords = generate_coordinate_grid(size=32)
            train_targets = generate_synthetic_2d_image(size=32, seed=seed + 10).reshape(-1, 1)
            eval_coords, eval_targets = train_coords, train_targets

        model_adam, model_muon, opt_adam, opt_muon = make_model_pair(
            "vanilla_mlp", seed, in_dim=in_dim
        )
        for _ in range(steps):
            train_step(model_adam, opt_adam, train_coords, train_targets)
            train_step(model_muon, opt_muon, train_coords, train_targets)

        with torch.no_grad():
            psnr_adam = compute_psnr(model_adam(eval_coords), eval_targets)
            psnr_muon = compute_psnr(model_muon(eval_coords), eval_targets)

        gain = psnr_muon - psnr_adam
        domain_results[domain] = {
            "psnr_adam": round(psnr_adam, 2),
            "psnr_muon": round(psnr_muon, 2),
            "psnr_gain_db": round(gain, 2),
            "improved": psnr_muon >= psnr_adam,
        }

    max_gain = max(v["psnr_gain_db"] for v in domain_results.values())
    all_improved = all(v["improved"] for v in domain_results.values())

    return {
        "domains": domain_results,
        "max_psnr_gain_db": round(max_gain, 2),
        "all_domains_improved": all_improved,
        "status": "verified" if all_improved else "unverified",
    }


def run_all_benchmarks() -> dict:
    """Run full evaluation harness for all 4 claims."""
    return {
        "claim1_stable_rank": run_claim1_stable_rank_degradation_test(),
        "claim2_image_overfitting": run_claim2_image_overfitting_test(),
        "claim3_sparse_ct": run_claim3_sparse_ct_test(),
        "claim4_multidomain": run_claim4_multidomain_extension_test(),
    }
