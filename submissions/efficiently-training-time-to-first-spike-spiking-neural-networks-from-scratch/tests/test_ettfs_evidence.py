import json
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ettfs_snn.ettfs import (
    ettfs_init,
    kaiming_init,
    evaluate_pooling_constraints,
    run_component_ablation,
    run_decoder_comparison_benchmark,
    run_init_signal_propagation_test,
    simulate_if_layers,
    ttfs_encode,
)
from ettfs_snn.evidence import CLAIMS, build_claim_results, generate_evidence_bundle

PAGE_NAMES = [
    "00-summary.md",
    "01-claim-1-init-signal-propagation.md",
    "02-claim-2-temporal-weighting-decoder.md",
    "03-claim-3-pooling-constraints.md",
    "04-claim-4-dataset-accuracies-not-reproduced.md",
    "05-claim-5-component-ablation.md",
]


def test_ettfs_init_scales_above_kaiming():
    fan_in = 128
    kaiming = kaiming_init(torch.empty(64, fan_in), fan_in=fan_in)
    ettfs = ettfs_init(torch.empty(64, fan_in), fan_in=fan_in, layer_index=4)
    assert ettfs.std() > kaiming.std()


def test_ttfs_encode_is_single_spike():
    intensity = torch.rand(8, 16)
    train = ttfs_encode(intensity, t_max=20)
    assert torch.all(train.sum(dim=0) == 1.0)


def test_simulate_if_layers_fires_each_neuron_at_most_once():
    train = ttfs_encode(torch.rand(8, 32), t_max=16)
    weights = [ettfs_init(torch.empty(32, 32), fan_in=32, layer_index=1)]
    out_train, stats = simulate_if_layers(train, weights, threshold=0.5)
    assert torch.all(out_train.sum(dim=0) <= 1.0)
    assert stats[0]["psc_std"] > 0.0


def test_claim1_signal_propagation_verified():
    res = run_init_signal_propagation_test()
    assert res["status"] == "verified"
    assert res["final_ettfs_firing_fraction"] > res["final_kaiming_firing_fraction"]
    assert res["ettfs_psc_std_decay_factor"] < res["kaiming_psc_std_decay_factor"]
    assert len(res["per_layer"]) == res["depth"]


def test_claim2_decoder_reduces_steps():
    metrics = run_decoder_comparison_benchmark()
    assert metrics["avg_twd_steps"] < metrics["avg_tq_steps"]
    assert metrics["overall_reduction_percent"] > 0


def test_claim3_pooling_commutation_measured():
    constraints = evaluate_pooling_constraints()
    assert constraints["avg_pool_commutation_error"] < 1e-5
    assert constraints["max_pool_commutation_error"] > 1e-3
    assert constraints["avg_pooling_preserves_single_spike"] is True
    assert constraints["max_pooling_preserves_single_spike"] is False


def test_claim5_ablation_trains_and_improves():
    ablation = run_component_ablation()
    assert ablation["improved"] is True
    assert ablation["accuracy_gain_full_vs_baseline"] > 0
    assert (
        ablation["full_ettfs_init_avgpool_norm"]
        > ablation["baseline_kaiming_maxpool_nonorm"]
    )


def test_claim_results_cover_every_live_claim():
    results = build_claim_results()
    assert [entry["claim"] for entry in results] == CLAIMS
    statuses = {entry["status"] for entry in results}
    assert "unreplicated" in statuses, "dataset-scale claim must stay unreplicated"


def test_determinism_of_benchmarks():
    assert run_decoder_comparison_benchmark() == run_decoder_comparison_benchmark()
    assert evaluate_pooling_constraints() == evaluate_pooling_constraints()


def test_evidence_bundle_and_pages(tmp_path):
    pages_dir = tmp_path / "pages"
    bundle_path = generate_evidence_bundle(tmp_path / "evidence", pages_dir)
    with open(bundle_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["paper_id"] == "3EcT46wsdc"
    assert len(data["target_claims"]) == len(CLAIMS)
    for name in PAGE_NAMES:
        assert (pages_dir / name).exists(), f"missing page {name}"


def test_committed_pages_are_complete():
    pages = project_root / "pages"
    assert sorted(p.name for p in pages.glob("*.md")) == PAGE_NAMES


def test_committed_pages_contain_computed_numbers():
    text = (project_root / "pages" / "01-claim-1-init-signal-propagation.md").read_text()
    assert "PSC std" in text
    ablation_text = (project_root / "pages" / "05-claim-5-component-ablation.md").read_text()
    assert "%" in ablation_text
