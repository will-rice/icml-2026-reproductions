import json
import math

from grokking_repro.evidence import (
    ATTEMPT_ID,
    UPSTREAM_REVISION,
    build_evidence_bundle,
    detect_grokking_metrics,
    run_hyperparameter_sweep,
)

EXPECTED_ATTEMPT_ID = "4b8e5145-c432-4786-ace1-6270e8a2e192"
EXPECTED_UPSTREAM_REVISION = "arxiv:2601.19791v4"


def test_detect_grokking_metrics_finds_delay_between_fit_and_generalization():
    metrics = detect_grokking_metrics(
        steps=[0, 1, 2, 3, 4, 5],
        train_losses=[1.0, 0.4, 0.08, 0.05, 0.04, 0.03],
        test_losses=[1.2, 1.1, 0.95, 0.8, 0.2, 0.1],
        train_threshold=0.1,
        test_threshold=0.25,
    )

    assert metrics["overfit_step"] == 2
    assert metrics["grokking_step"] == 4
    assert metrics["delay_steps"] == 2


def test_hyperparameter_sweep_returns_finite_ordered_results():
    rows = run_hyperparameter_sweep(
        sample_sizes=[8, 14],
        weight_decays=[0.03, 0.12],
        steps=120,
        seed=7,
    )

    assert len(rows) == 4
    assert {row["sample_size"] for row in rows} == {8, 14}
    assert {row["weight_decay"] for row in rows} == {0.03, 0.12}
    assert all(math.isfinite(row["final_train_loss"]) for row in rows)
    assert all(math.isfinite(row["final_test_loss"]) for row in rows)


def test_bundle_records_current_attempt_and_claim_boundaries():
    rows = [
        {
            "sample_size": 8,
            "weight_decay": 0.03,
            "overfit_step": 10,
            "grokking_step": 90,
            "delay_steps": 80,
            "final_train_loss": 0.01,
            "final_test_loss": 0.2,
        },
        {
            "sample_size": 14,
            "weight_decay": 0.12,
            "overfit_step": 14,
            "grokking_step": 44,
            "delay_steps": 30,
            "final_train_loss": 0.02,
            "final_test_loss": 0.18,
        },
    ]

    bundle = build_evidence_bundle(rows)

    assert ATTEMPT_ID == EXPECTED_ATTEMPT_ID
    assert UPSTREAM_REVISION == EXPECTED_UPSTREAM_REVISION
    assert bundle["attempt_id"] == EXPECTED_ATTEMPT_ID
    assert bundle["upstream_revision"] == EXPECTED_UPSTREAM_REVISION
    assert bundle["claims"][0]["status"] == "paper-audit"
    assert bundle["claims"][2]["status"] in {"toy", "inconclusive"}
    assert bundle["claims"][3]["status"] == "unreplicated"


def test_run_evidence_writes_machine_readable_files(tmp_path):
    from grokking_repro.evidence import run_evidence

    bundle = run_evidence(output_dir=tmp_path / "evidence", steps=40)
    results = json.loads((tmp_path / "evidence" / "results.json").read_text())
    provenance = json.loads((tmp_path / "evidence" / "provenance.json").read_text())

    assert results == bundle
    assert provenance["attempt_id"] == EXPECTED_ATTEMPT_ID
    assert provenance["paid_api_cost_usd"] == 0.0
