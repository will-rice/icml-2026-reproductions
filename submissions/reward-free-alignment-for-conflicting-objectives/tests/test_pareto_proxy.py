import json
from pathlib import Path

import pytest
import torch

from reward_free_alignment.pareto_proxy import (
    DEFAULT_RADIUS,
    PARAM_DIM,
    SEED,
    hypervolume,
    make_objective_data,
    make_reward_directions,
    train_policy,
    validation_accuracy,
)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).parent.parent


def test_reward_directions_conflict():
    r1, r2 = make_reward_directions()
    assert torch.linalg.vector_norm(r1).item() == pytest.approx(1.0)
    assert torch.linalg.vector_norm(r2).item() == pytest.approx(1.0)
    assert torch.dot(r1, r2).item() == pytest.approx(-0.3, abs=1e-6)


def test_hypervolume_single_and_frontier():
    assert hypervolume([(0.7, 0.695)]) == round(0.7 * 0.695, 6)
    assert hypervolume([(1.0, 0.5), (0.5, 1.0)]) == round(1.0 * 0.5 + 0.5 * 0.5, 6)
    dominated_point_is_ignored = hypervolume([(1.0, 0.5), (0.9, 0.4)])
    assert dominated_point_is_ignored == round(1.0 * 0.5, 6)


def test_short_training_is_deterministic(monkeypatch):
    import reward_free_alignment.pareto_proxy as pp

    monkeypatch.setattr(pp, "TRAIN_STEPS", 10)
    outcomes = []
    for _ in range(2):
        generator = torch.Generator().manual_seed(SEED)
        r1, r2 = make_reward_directions()
        objectives = (
            make_objective_data(r1, generator),
            make_objective_data(r2, generator),
        )
        theta0 = 0.3 * torch.randn(PARAM_DIM, generator=generator)
        theta, losses = train_policy(
            objectives, theta0, torch.tensor([0.5, 0.5]), DEFAULT_RADIUS, True
        )
        outcomes.append(
            (losses, validation_accuracy(objectives[0], theta))
        )
    assert outcomes[0] == outcomes[1]


def test_pareto_page_matches_evidence(project_root):
    evidence = json.loads((project_root / "evidence/results.json").read_text("utf-8"))
    audit = evidence["audits"]["pareto_proxy"]
    page = (project_root / "pages/06-pareto-proxy.md").read_text("utf-8")
    assert str(audit["raco_hypervolume"]) in page
    assert str(audit["baseline_hypervolume"]) in page
    assert f"{audit['raco_dominates_baseline_count']} of {audit['weight_settings']}" in page
    for row in audit["frontier"]:
        assert str(row["raco_val_acc"][0]) in page
        assert str(row["baseline_val_acc"][0]) in page
    for row in audit["ablation"]:
        assert str(row["min_val_acc"]) in page
