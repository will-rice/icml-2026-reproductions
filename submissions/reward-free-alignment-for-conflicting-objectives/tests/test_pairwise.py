import math
import pytest
import torch
from torch import tensor
from torch.nn.functional import logsigmoid
from reward_free_alignment.pairwise import (
    PairwiseBatch,
    pairwise_logistic_loss,
    objective_losses,
    objective_gradients,
)


def test_pairwise_loss_matches_closed_form():
    batch = PairwiseBatch(
        tensor([-0.2]), tensor([-0.8]), tensor([-0.4]), tensor([-0.6])
    )
    loss = pairwise_logistic_loss(batch, beta=0.5)
    assert torch.allclose(loss, -logsigmoid(tensor([0.2])).mean())


def test_objective_losses_are_not_scalarized():
    fixture_a = PairwiseBatch(tensor([-0.1]), tensor([-0.9]), tensor([-0.3]), tensor([-0.7]))
    fixture_b = PairwiseBatch(tensor([-0.5]), tensor([-0.4]), tensor([-0.2]), tensor([-0.3]))
    losses = objective_losses((fixture_a, fixture_b), beta=0.5)
    assert losses.shape == (2,)
    assert not torch.equal(losses[0], losses[1])


def test_invalid_beta_raises_value_error():
    batch = PairwiseBatch(tensor([-0.2]), tensor([-0.8]), tensor([-0.4]), tensor([-0.6]))
    with pytest.raises(ValueError, match="beta"):
        pairwise_logistic_loss(batch, beta=0.0)
    with pytest.raises(ValueError, match="beta"):
        pairwise_logistic_loss(batch, beta=-1.0)
    with pytest.raises(ValueError, match="beta"):
        pairwise_logistic_loss(batch, beta=float("nan"))


def test_shape_mismatch_raises_value_error():
    batch = PairwiseBatch(tensor([-0.2, -0.3]), tensor([-0.8]), tensor([-0.4]), tensor([-0.6]))
    with pytest.raises(ValueError, match="shape"):
        pairwise_logistic_loss(batch, beta=0.5)


def test_objective_gradients_flattens_and_checks_finite():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
    batch1 = PairwiseBatch(param[0:1], param[1:2], tensor([-0.4]), tensor([-0.6]))
    batch2 = PairwiseBatch(param[1:2], param[0:1], tensor([-0.2]), tensor([-0.3]))
    losses = objective_losses((batch1, batch2), beta=1.0)
    grads = objective_gradients(losses, (param,))
    assert len(grads) == 2
    assert grads[0].shape == (2,)
    assert grads[1].shape == (2,)
    assert torch.isfinite(grads[0]).all()
    assert torch.isfinite(grads[1]).all()
