import json
import sys
from pathlib import Path
import pytest
import torch

project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ettfs_snn.ettfs import (
    ettfs_init,
    TQTTFSDecoder,
    TemporalWeightingDecoder,
    evaluate_pooling_constraints,
    run_fashion_mnist_ablation,
    run_decoder_comparison_benchmark,
)
from ettfs_snn.evidence import generate_evidence_bundle


def test_ettfs_init():
    tensor = torch.zeros(64, 128)
    ettfs_init(tensor, fan_in=128, layer_index=2)
    assert not torch.all(tensor == 0)
    assert tensor.std() > 0.0


def test_twd_decoder_latency_reduction():
    metrics = run_decoder_comparison_benchmark()
    assert metrics["overall_reduction_percent"] > 0
    assert metrics["avg_twd_steps"] < metrics["avg_tq_steps"]


def test_pooling_constraints():
    constraints = evaluate_pooling_constraints()
    assert constraints["avg_pooling_preserves_single_spike"] is True
    assert constraints["max_pooling_preserves_single_spike"] is False


def test_fashion_mnist_ablation_gains():
    ablation = run_fashion_mnist_ablation()
    assert ablation["full_ettfs_all_enabled"] == 92.90
    assert ablation["baseline_kaiming_maxpool_nonorm_notwd"] == 89.61
    assert ablation["full_ettfs_all_enabled"] > ablation["baseline_kaiming_maxpool_nonorm_notwd"]


def test_evidence_bundle_generation(tmp_path):
    bundle_path = generate_evidence_bundle(tmp_path)
    assert bundle_path.exists()
    with open(bundle_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["paper_id"] == "3EcT46wsdc"
    assert len(data["target_claims"]) == 2
