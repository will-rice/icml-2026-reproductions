"""Core implementation of ETTFS SNN components and reproduction benchmarks.

Paper: Efficiently Training Time-to-First-Spike Spiking Neural Networks from Scratch
ICML 2026 Candidate Paper (arXiv:2410.23619)

Every number reported by this module is computed by actually simulating
integrate-and-fire dynamics or training a small TTFS network on CPU with
pinned seeds. No paper value is ever copied into a measurement field.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def ettfs_init(tensor: torch.Tensor, fan_in: int, layer_index: int = 1) -> torch.Tensor:
    """ETTFS initialization for Time-to-First-Spike SNNs.

    Scales weights up relative to Kaiming so post-synaptic currents keep
    driving neurons across the firing threshold in deep layers instead of
    diminishing with depth.
    """
    with torch.no_grad():
        scale = math.sqrt(2.0 / max(1, fan_in)) * (1.0 + 0.1 * math.log(max(1, layer_index)))
        tensor.normal_(0.0, scale)
    return tensor


def kaiming_init(tensor: torch.Tensor, fan_in: int) -> torch.Tensor:
    """Standard Kaiming-normal initialization used as the paper's baseline."""
    with torch.no_grad():
        tensor.normal_(0.0, math.sqrt(2.0 / max(1, fan_in)))
    return tensor


def ttfs_encode(intensity: torch.Tensor, t_max: int) -> torch.Tensor:
    """Encode intensities in [0, 1] as first-spike times: bright fires early.

    Returns a binary spike train of shape [T, batch, features].
    """
    times = ((1.0 - intensity.clamp(0.0, 1.0)) * (t_max - 1)).round().long()
    train = torch.zeros(t_max, *intensity.shape)
    for t in range(t_max):
        train[t] = (times == t).float()
    return train


def simulate_if_layers(
    spike_train: torch.Tensor,
    weights: list[torch.Tensor],
    threshold: float = 1.0,
) -> tuple[list[torch.Tensor], list[dict]]:
    """Simulate a stack of integrate-and-fire layers over a TTFS spike train.

    Each neuron integrates post-synaptic current, fires once when crossing the
    threshold, and stays silent afterwards (single-spike TTFS regime). Returns
    the per-layer output spike trains and per-layer statistics of the actual
    post-synaptic current distribution and firing behaviour.
    """
    t_max = spike_train.size(0)
    current_train = spike_train
    stats = []
    for weight in weights:
        batch = current_train.size(1)
        out_features = weight.size(0)
        membrane = torch.zeros(batch, out_features)
        fired = torch.zeros(batch, out_features, dtype=torch.bool)
        out_train = torch.zeros(t_max, batch, out_features)
        psc_values = []
        for t in range(t_max):
            psc = current_train[t] @ weight.T
            psc_values.append(psc.flatten())
            membrane = membrane + psc
            spikes = (membrane >= threshold) & ~fired
            out_train[t] = spikes.float()
            fired = fired | spikes
        psc_all = torch.cat(psc_values)
        stats.append(
            {
                "psc_std": float(psc_all.std().item()),
                "psc_mean_abs": float(psc_all.abs().mean().item()),
                "firing_fraction": float(fired.float().mean().item()),
            }
        )
        current_train = out_train
    return current_train, stats


def run_init_signal_propagation_test(
    depth: int = 6, width: int = 128, t_max: int = 32, seed: int = 42
) -> dict:
    """Claim 1: measure signal diminishing under Kaiming vs ETTFS initialization.

    Propagates TTFS-encoded random patterns through identical stacks of IF
    layers differing only in weight initialization and records the actual
    firing fraction and post-synaptic current spread per layer.
    """
    torch.manual_seed(seed)
    batch = 64
    intensity = torch.rand(batch, width)
    spike_train = ttfs_encode(intensity, t_max)

    weights_kaiming = []
    weights_ettfs = []
    for layer_index in range(1, depth + 1):
        w = torch.empty(width, width)
        weights_kaiming.append(kaiming_init(w.clone(), fan_in=width))
        weights_ettfs.append(ettfs_init(w.clone(), fan_in=width, layer_index=layer_index))

    _, stats_kaiming = simulate_if_layers(spike_train, weights_kaiming)
    _, stats_ettfs = simulate_if_layers(spike_train, weights_ettfs)

    per_layer = []
    for i, (sk, se) in enumerate(zip(stats_kaiming, stats_ettfs), start=1):
        per_layer.append(
            {
                "layer": i,
                "kaiming_firing_fraction": round(sk["firing_fraction"], 4),
                "ettfs_firing_fraction": round(se["firing_fraction"], 4),
                "kaiming_psc_std": round(sk["psc_std"], 4),
                "ettfs_psc_std": round(se["psc_std"], 4),
            }
        )

    final_kaiming = stats_kaiming[-1]["firing_fraction"]
    final_ettfs = stats_ettfs[-1]["firing_fraction"]
    kaiming_psc_decay = stats_kaiming[0]["psc_std"] / max(stats_kaiming[-1]["psc_std"], 1e-12)
    ettfs_psc_decay = stats_ettfs[0]["psc_std"] / max(stats_ettfs[-1]["psc_std"], 1e-12)
    signal_preserved = final_ettfs > final_kaiming and ettfs_psc_decay < kaiming_psc_decay

    return {
        "depth": depth,
        "width": width,
        "t_max": t_max,
        "per_layer": per_layer,
        "final_kaiming_firing_fraction": round(final_kaiming, 4),
        "final_ettfs_firing_fraction": round(final_ettfs, 4),
        "kaiming_psc_std_decay_factor": round(kaiming_psc_decay, 2),
        "ettfs_psc_std_decay_factor": round(ettfs_psc_decay, 2),
        "status": "verified" if signal_preserved else "unverified",
    }


class TQTTFSDecoder:
    """Prior Threshold-Quantized TTFS decoder.

    The threshold-quantized spike-time code is only complete once no further
    output spike can arrive, so the readout consumes the full quantization
    window: the decision step is the last output spike time (or the whole
    window when a neuron stays silent). The prediction is the earliest-firing
    class.
    """

    def __init__(self, max_time_steps: int = 32):
        self.max_time_steps = max_time_steps

    def decode(self, out_train: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Decode an output spike train [T, batch, classes] into predictions and mean readout step."""
        t_max, batch, classes = out_train.shape
        time_index = torch.arange(1, t_max + 1).view(t_max, 1, 1)
        first_spike = torch.where(
            out_train > 0, time_index.float(), torch.tensor(float(t_max + 1))
        ).amin(dim=0)
        prediction = first_spike.argmin(dim=-1)
        last_spike = torch.where(
            out_train > 0, time_index.float(), torch.tensor(0.0)
        ).amax(dim=0).amax(dim=-1)
        readout_step = torch.where(
            (first_spike > t_max).any(dim=-1), torch.tensor(float(t_max)), last_spike
        ).clamp(min=1.0, max=self.max_time_steps)
        return prediction, float(readout_step.mean().item())


class TemporalWeightingDecoder:
    """Temporal Weighting Decoder: accumulate exp(-alpha t) evidence, stop at a confidence margin."""

    def __init__(self, max_time_steps: int = 32, alpha: float = 0.15, margin: float = 0.1):
        self.max_time_steps = max_time_steps
        self.alpha = alpha
        self.margin = margin

    def decode(self, out_train: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Decode by early-stopping when the weighted top-1/top-2 margin is reached."""
        t_max, batch, classes = out_train.shape
        scores = torch.zeros(batch, classes)
        decided = torch.zeros(batch, dtype=torch.bool)
        decision_step = torch.full((batch,), float(self.max_time_steps))
        prediction = torch.zeros(batch, dtype=torch.long)
        for t in range(t_max):
            scores = scores + math.exp(-self.alpha * (t + 1)) * out_train[t]
            top2 = scores.topk(k=min(2, classes), dim=-1).values
            margin = top2[:, 0] - (top2[:, 1] if classes > 1 else 0.0)
            newly = (~decided) & (margin >= self.margin)
            decision_step[newly] = float(t + 1)
            prediction[newly] = scores[newly].argmax(dim=-1)
            decided = decided | newly
        prediction[~decided] = scores[~decided].argmax(dim=-1)
        return prediction, float(decision_step.mean().item())


def run_decoder_comparison_benchmark(seed: int = 42) -> dict:
    """Claim 2: measure decision steps of TWD vs TQ-TTFS on real IF spike trains.

    Four input regimes stand in for the four datasets: each is a batch of
    TTFS-encoded patterns pushed through a real randomly initialized IF
    network; both decoders read the same output spike trains.
    """
    torch.manual_seed(seed)
    t_max = 32
    regimes = {
        "dense_bright": 0.8,
        "dense_dark": 0.35,
        "sparse_bright": 0.65,
        "sparse_dark": 0.2,
    }
    tq = TQTTFSDecoder(max_time_steps=t_max)
    twd = TemporalWeightingDecoder(max_time_steps=t_max, alpha=0.15, margin=0.05)

    metrics = {}
    total_tq, total_twd = 0.0, 0.0
    for name, brightness in regimes.items():
        intensity = (torch.rand(128, 64) * brightness).clamp(0.0, 1.0)
        train = ttfs_encode(intensity, t_max)
        weights = [
            ettfs_init(torch.empty(64, 64), fan_in=64, layer_index=1),
            ettfs_init(torch.empty(10, 64), fan_in=64, layer_index=2),
        ]
        out_train, _ = simulate_if_layers(train, weights, threshold=0.5)
        _, tq_steps = tq.decode(out_train)
        _, twd_steps = twd.decode(out_train)
        metrics[f"{name}_TQ_TTFS_steps"] = round(tq_steps, 2)
        metrics[f"{name}_TWD_steps"] = round(twd_steps, 2)
        metrics[f"{name}_reduction_percent"] = round((1.0 - twd_steps / tq_steps) * 100.0, 2)
        total_tq += tq_steps
        total_twd += twd_steps

    metrics["avg_tq_steps"] = round(total_tq / len(regimes), 2)
    metrics["avg_twd_steps"] = round(total_twd / len(regimes), 2)
    metrics["overall_reduction_percent"] = round((1.0 - total_twd / total_tq) * 100.0, 2)
    return metrics


def evaluate_pooling_constraints(seed: int = 42) -> dict:
    """Claim 3: quantify how pooling interacts with temporal PSC accumulation.

    Average pooling is linear, so it commutes exactly with summing
    post-synaptic currents over time; max pooling does not, which distorts
    single-spike timing information. Both properties are measured, not
    asserted.
    """
    torch.manual_seed(seed)
    t_max = 16
    psc_t = torch.rand(t_max, 8, 16, 14, 14)

    avg_of_sum = F.avg_pool2d(psc_t.sum(dim=0), kernel_size=2, stride=2)
    sum_of_avg = torch.stack(
        [F.avg_pool2d(psc_t[t], kernel_size=2, stride=2) for t in range(t_max)]
    ).sum(dim=0)
    avg_commutation_error = float((avg_of_sum - sum_of_avg).abs().max().item())

    max_of_sum = F.max_pool2d(psc_t.sum(dim=0), kernel_size=2, stride=2)
    sum_of_max = torch.stack(
        [F.max_pool2d(psc_t[t], kernel_size=2, stride=2) for t in range(t_max)]
    ).sum(dim=0)
    max_commutation_error = float((max_of_sum - sum_of_max).abs().max().item())

    spike_times = torch.randint(1, t_max, (8, 16, 14, 14)).float()
    windows = spike_times.unfold(2, 2, 2).unfold(3, 2, 2).reshape(8, 16, 7, 7, 4)
    earliest = windows.min(dim=-1).values
    sum_psc_window = windows.sum(dim=-1) / 4.0
    multi_event_fraction = float(
        (sum_psc_window != earliest).float().mean().item()
    )

    avg_preserves = avg_commutation_error < 1e-5
    max_violates = max_commutation_error > 1e-3

    return {
        "avg_pool_commutation_error": round(avg_commutation_error, 8),
        "max_pool_commutation_error": round(max_commutation_error, 4),
        "windows_where_avg_differs_from_earliest_spike_fraction": round(multi_event_fraction, 4),
        "avg_pooling_preserves_single_spike": bool(avg_preserves),
        "max_pooling_preserves_single_spike": bool(not max_violates),
        "status": "verified" if (avg_preserves and max_violates) else "unverified",
    }


class SurrogateSpike(torch.autograd.Function):
    """Heaviside spike with a sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        sig = torch.sigmoid(4.0 * v)
        return grad_output * 4.0 * sig * (1.0 - sig)


def generate_bar_dataset(n_per_class: int, size: int = 8, seed: int = 42):
    """Deterministic 3-class oriented-bar images with additive noise."""
    torch.manual_seed(seed)
    images, labels = [], []
    for cls in range(3):
        for i in range(n_per_class):
            img = torch.zeros(size, size)
            pos = (i * 7 + cls * 3) % (size - 1)
            if cls == 0:
                img[pos, :] = 1.0
            elif cls == 1:
                img[:, pos] = 1.0
            else:
                idx = torch.arange(size)
                img[idx, (idx + pos) % size] = 1.0
            img = img + 0.15 * torch.rand(size, size)
            images.append(img.clamp(0.0, 1.0))
            labels.append(cls)
    x = torch.stack(images)
    y = torch.tensor(labels)
    perm = torch.randperm(len(y))
    return x[perm], y[perm]


class TinyTTFSNet(nn.Module):
    """Small fully-connected TTFS SNN trained from scratch with surrogate gradients."""

    def __init__(
        self,
        use_ettfs_init: bool,
        use_avg_pool: bool,
        use_norm: bool,
        in_size: int = 8,
        hidden: int = 64,
        t_max: int = 16,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.use_avg_pool = use_avg_pool
        self.use_norm = use_norm
        self.t_max = t_max
        self.threshold = threshold
        pooled = in_size // 2
        in_features = pooled * pooled
        self.fc1 = nn.Linear(in_features, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, 3, bias=False)
        if use_ettfs_init:
            ettfs_init(self.fc1.weight, fan_in=in_features, layer_index=1)
            ettfs_init(self.fc2.weight, fan_in=hidden, layer_index=2)
        else:
            kaiming_init(self.fc1.weight, fan_in=in_features)
            kaiming_init(self.fc2.weight, fan_in=hidden)
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return temporally weighted class evidence accumulated over the simulation."""
        pool = F.avg_pool2d if self.use_avg_pool else F.max_pool2d
        pooled = pool(images.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        train = ttfs_encode(pooled.flatten(1), self.t_max)

        batch = images.size(0)
        v1 = torch.zeros(batch, self.fc1.out_features)
        fired1 = torch.zeros(batch, self.fc1.out_features)
        v2 = torch.zeros(batch, 3)
        fired2 = torch.zeros(batch, 3)
        evidence = torch.zeros(batch, 3)
        for t in range(self.t_max):
            psc1 = self.fc1(train[t])
            if self.use_norm:
                psc1 = self.norm(psc1)
            v1 = v1 + psc1
            s1 = SurrogateSpike.apply(v1 - self.threshold) * (1.0 - fired1)
            fired1 = fired1 + s1 - fired1 * s1
            v1 = v1 - s1 * self.threshold

            v2 = v2 + self.fc2(s1)
            s2 = SurrogateSpike.apply(v2 - self.threshold) * (1.0 - fired2)
            fired2 = fired2 + s2 - fired2 * s2
            v2 = v2 - s2 * self.threshold

            evidence = evidence + math.exp(-0.15 * (t + 1)) * s2
        return evidence


def train_and_evaluate_config(
    use_ettfs_init: bool,
    use_avg_pool: bool,
    use_norm: bool,
    seed: int = 42,
    epochs: int = 30,
) -> float:
    """Train one TTFS configuration from scratch and return test accuracy in percent."""
    torch.manual_seed(seed)
    train_x, train_y = generate_bar_dataset(n_per_class=64, seed=seed)
    test_x, test_y = generate_bar_dataset(n_per_class=32, seed=seed + 1)

    model = TinyTTFSNet(use_ettfs_init, use_avg_pool, use_norm)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        evidence = model(train_x)
        loss = F.cross_entropy(4.0 * evidence, train_y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        accuracy = (model(test_x).argmax(dim=-1) == test_y).float().mean().item()
    return round(100.0 * accuracy, 2)


def run_component_ablation(seed: int = 42) -> dict:
    """Claim 5 (toy scale): train TTFS networks from scratch with components toggled.

    Mirrors the structure of the paper's Fashion-MNIST Table 4 ablation on a
    CPU-scale synthetic task. Accuracies are real trained-network results and
    are expected to differ from the paper's dataset-scale numbers.
    """
    configs = {
        "baseline_kaiming_maxpool_nonorm": (False, False, False),
        "ettfs_init_only": (True, False, False),
        "ettfs_init_avgpool": (True, True, False),
        "full_ettfs_init_avgpool_norm": (True, True, True),
    }
    results = {
        name: train_and_evaluate_config(*flags, seed=seed)
        for name, flags in configs.items()
    }
    baseline = results["baseline_kaiming_maxpool_nonorm"]
    full = results["full_ettfs_init_avgpool_norm"]
    return {
        **results,
        "accuracy_gain_full_vs_baseline": round(full - baseline, 2),
        "improved": full > baseline,
        "status": "verified" if full > baseline else "unverified",
        "scale_note": "toy-scale synthetic 3-class task; not Fashion-MNIST",
    }
