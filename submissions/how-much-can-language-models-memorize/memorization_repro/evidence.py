from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

ATTEMPT_ID = "8de87cc9-1d39-49a6-b552-4b4dd7e67e0e"
PAPER_ID = "bA6BgSbaUi"
UPSTREAM_REVISION = "arxiv:2505.24832v3"
TARGET_CLAIMS = [
    "GPT-style transformers trained on uniform random data show an empirical "
    "memorization-capacity plateau of about 3.6 bits per parameter (Figure 1)",
    "Capacity estimates across model widths and depths support a roughly "
    "linear bits-per-parameter scaling law, with bfloat16 to float32 "
    "increasing capacity only modestly (Table 1)",
]


def memorized_bits_from_nll(
    *, mean_nll_bits_per_token: list[float], sequence_length: int, vocab_size: int
) -> float:
    baseline_bits_per_token = math.log2(vocab_size)
    return sum(
        max(0.0, baseline_bits_per_token - nll_bits) * sequence_length
        for nll_bits in mean_nll_bits_per_token
    )


def generate_uniform_token_sequences(
    *, num_sequences: int, sequence_length: int, vocab_size: int, seed: int
) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        [rng.randrange(vocab_size) for _ in range(sequence_length)]
        for _ in range(num_sequences)
    ]


@dataclass(frozen=True)
class TinyModelSpec:
    name: str
    d_model: int
    n_layers: int
    n_heads: int


class TinyCausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        sequence_length: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(sequence_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device).expand(batch, length)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.blocks(hidden, mask=mask)
        return self.head(self.norm(hidden))


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _normalise_model_specs(model_specs: list[dict[str, Any]]) -> list[TinyModelSpec]:
    return [
        TinyModelSpec(
            name=str(spec["name"]),
            d_model=int(spec["d_model"]),
            n_layers=int(spec["n_layers"]),
            n_heads=int(spec["n_heads"]),
        )
        for spec in model_specs
    ]


def _train_one_experiment(
    *,
    spec: TinyModelSpec,
    dataset_size: int,
    sequence_length: int,
    vocab_size: int,
    train_steps: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    raw_sequences = generate_uniform_token_sequences(
        num_sequences=dataset_size,
        sequence_length=sequence_length + 1,
        vocab_size=vocab_size,
        seed=seed,
    )
    data = torch.tensor(raw_sequences, dtype=torch.long)
    inputs = data[:, :-1]
    targets = data[:, 1:]

    model = TinyCausalTransformer(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        d_model=spec.d_model,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)

    for _ in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(inputs)
        token_losses = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape(dataset_size, sequence_length)
        example_nll_bits = (token_losses.sum(dim=1) / math.log(2)) / sequence_length
        mean_nll_bits = float(example_nll_bits.mean().item())
        memorized_bits = memorized_bits_from_nll(
            mean_nll_bits_per_token=example_nll_bits.tolist(),
            sequence_length=sequence_length,
            vocab_size=vocab_size,
        )
    parameter_count = _parameter_count(model)
    return {
        "model_name": spec.name,
        "dataset_size": dataset_size,
        "sequence_length": sequence_length,
        "vocab_size": vocab_size,
        "train_steps": train_steps,
        "parameter_count": parameter_count,
        "mean_train_nll_bits_per_token": mean_nll_bits,
        "memorized_bits": float(memorized_bits),
        "bits_per_parameter": float(memorized_bits / parameter_count),
    }


def build_evidence_bundle(
    *,
    attempt_id: str,
    paper_id: str,
    upstream_revision: str,
    experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    has_multiple_sizes = len({row["parameter_count"] for row in experiments}) >= 2
    has_multiple_datasets = len({row["dataset_size"] for row in experiments}) >= 2
    claim_1_status = "toy" if has_multiple_datasets else "inconclusive"
    claim_2_status = "toy" if has_multiple_sizes else "inconclusive"
    return {
        "attempt_id": attempt_id,
        "paper_id": paper_id,
        "title": "How much can language models memorize?",
        "upstream_revision": upstream_revision,
        "measurements": experiments,
        "paper_reported_context": {
            "plateau_bits_per_parameter": 3.6,
            "context_only": True,
            "source": upstream_revision,
        },
        "claims": [
            {
                "claim": TARGET_CLAIMS[0],
                "status": claim_1_status,
                "evidence": (
                    "Tiny CPU GPT-style models were trained on seeded uniform "
                    "random token data. The measured bits-per-parameter values "
                    "are toy-scale observations and are not the paper-reported "
                    "3.6 bits-per-parameter plateau."
                ),
            },
            {
                "claim": TARGET_CLAIMS[1],
                "status": claim_2_status,
                "evidence": (
                    "The run compares tiny model sizes under the same seeded "
                    "random-data protocol. Precision effects are not reproduced "
                    "because deterministic CPU bfloat16 training was not used."
                ),
            },
        ],
        "provenance": {
            "attempt_id": attempt_id,
            "paper_id": paper_id,
            "upstream_revision": upstream_revision,
            "seed_policy": "Python random and torch manual seeds are fixed per run.",
            "paid_api_cost_usd": 0.0,
        },
    }


def _write_summary(bundle: dict[str, Any], output_dir: Path) -> None:
    pages_dir = output_dir.parent / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Memorization-capacity reproduction",
        "",
        "This submission runs a CPU-only toy reimplementation of the paper's",
        "uniform-random-data memorization protocol. It does not reproduce the",
        "full 500K-to-1.5B parameter sweep or treat paper-reported values as",
        "measurements.",
        "",
        "## Claim status",
        "",
    ]
    for claim in bundle["claims"]:
        lines.append(f"- `{claim['status']}`: {claim['claim']}")
    lines.extend(
        [
            "",
            "## Measurements",
            "",
            "| model | dataset | params | memorized bits | bits/param |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in bundle["measurements"]:
        lines.append(
            "| {model_name} | {dataset_size} | {parameter_count} | "
            "{memorized_bits:.6f} | {bits_per_parameter:.6f} |".format(**row)
        )
    (pages_dir / "00-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evidence(
    *,
    output_dir: str | Path,
    train_steps: int = 2,
    dataset_sizes: list[int] | None = None,
    model_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_sizes = dataset_sizes or [8, 16]
    model_specs = model_specs or [
        {"name": "tiny-8", "d_model": 8, "n_layers": 1, "n_heads": 2},
        {"name": "tiny-16", "d_model": 16, "n_layers": 1, "n_heads": 2},
    ]
    experiments = []
    for spec_index, spec in enumerate(_normalise_model_specs(model_specs)):
        for dataset_size in dataset_sizes:
            experiments.append(
                _train_one_experiment(
                    spec=spec,
                    dataset_size=dataset_size,
                    sequence_length=8,
                    vocab_size=16,
                    train_steps=train_steps,
                    seed=20260729 + spec_index * 1000 + dataset_size,
                )
            )
    bundle = build_evidence_bundle(
        attempt_id=ATTEMPT_ID,
        paper_id=PAPER_ID,
        upstream_revision=UPSTREAM_REVISION,
        experiments=experiments,
    )
    (output_path / "results.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_path / "provenance.json").write_text(
        json.dumps(bundle["provenance"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_path / "bundle.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(bundle, output_path)
    return bundle
