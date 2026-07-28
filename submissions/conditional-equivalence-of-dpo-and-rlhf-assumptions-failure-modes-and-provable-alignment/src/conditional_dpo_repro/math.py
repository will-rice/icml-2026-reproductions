from dataclasses import dataclass
import math

PROBABILITY_TOLERANCE = 1e-12


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def softplus(value: float) -> float:
    value = _finite("softplus input", value)
    if value > 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def sigmoid(value: float) -> float:
    value = _finite("sigmoid input", value)
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def logit(p: float) -> float:
    p = _finite("p", p)
    if not 0.0 < p < 1.0:
        raise ValueError("probability p must lie in (0, 1)")
    return math.log(p) - math.log1p(-p)


@dataclass(frozen=True)
class TwoResponsePolicy:
    preferred: float
    dispreferred: float

    def __post_init__(self):
        preferred = _finite("preferred", self.preferred)
        dispreferred = _finite("dispreferred", self.dispreferred)
        if not 0.0 < preferred < 1.0 or not 0.0 < dispreferred < 1.0:
            raise ValueError("policy probabilities must lie in (0, 1)")
        if abs(preferred + dispreferred - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError("policy probabilities must sum to one")

    @property
    def delta(self) -> float:
        return math.log(self.preferred) - math.log(self.dispreferred)


def policy_from_delta(delta: float) -> TwoResponsePolicy:
    delta = _finite("delta", delta)
    p_win = sigmoid(delta)
    p_lose = sigmoid(-delta)
    # Ensure sum is exactly 1.0 numerically
    p_lose = 1.0 - p_win
    return TwoResponsePolicy(p_win, p_lose)


def _loss_inputs(delta: float, delta_ref: float, beta: float):
    delta = _finite("delta", delta)
    delta_ref = _finite("delta_ref", delta_ref)
    beta = _finite("beta", beta)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    return delta, delta_ref, beta


def rlhf_optimal_delta(delta_ref: float, reward_gap: float, beta: float) -> float:
    delta_ref = _finite("delta_ref", delta_ref)
    reward_gap = _finite("reward_gap", reward_gap)
    beta = _finite("beta", beta)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    return delta_ref + reward_gap / beta


def dpo_loss(delta: float, delta_ref: float, beta: float) -> float:
    delta, delta_ref, beta = _loss_inputs(delta, delta_ref, beta)
    return softplus(-beta * (delta - delta_ref))


def dpo_loss_derivative(delta: float, delta_ref: float, beta: float) -> float:
    delta, delta_ref, beta = _loss_inputs(delta, delta_ref, beta)
    return -beta * sigmoid(-beta * (delta - delta_ref))


def bt_population_loss(
    delta: float, delta_ref: float, reward_gap: float, beta: float
) -> float:
    delta, delta_ref, beta = _loss_inputs(delta, delta_ref, beta)
    reward_gap = _finite("reward_gap", reward_gap)
    q = sigmoid(reward_gap)
    model_logit = beta * (delta - delta_ref)
    return q * softplus(-model_logit) + (1.0 - q) * softplus(model_logit)


def constrained_rlhf_objective(
    policy: TwoResponsePolicy,
    reference: TwoResponsePolicy,
    reward_gap: float,
    beta: float,
    gamma: float,
) -> float:
    reward_gap = _finite("reward_gap", reward_gap)
    beta = _finite("beta", beta)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    gamma = _finite("gamma", gamma)
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")

    p = policy.preferred
    p_ref = reference.preferred
    q = policy.dispreferred
    q_ref = reference.dispreferred

    kl = p * (math.log(p) - math.log(p_ref)) + q * (math.log(q) - math.log(q_ref))
    return p * reward_gap - beta * kl + gamma * policy.delta


def solve_exact_constrained_rlhf(
    reference: TwoResponsePolicy,
    reward_gap: float,
    beta: float,
    gamma: float,
) -> dict[str, object]:
    gamma = _finite("gamma", gamma)
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")

    if gamma == 0.0:
        opt_delta = rlhf_optimal_delta(reference.delta, reward_gap, beta)
        opt_policy = policy_from_delta(opt_delta)
        obj = constrained_rlhf_objective(opt_policy, reference, reward_gap, beta, 0.0)

        # Numerical derivative check
        h = 1e-6
        obj_plus = constrained_rlhf_objective(
            policy_from_delta(opt_delta + h), reference, reward_gap, beta, 0.0
        )
        obj_minus = constrained_rlhf_objective(
            policy_from_delta(opt_delta - h), reference, reward_gap, beta, 0.0
        )
        first_order_residual = abs((obj_plus - obj_minus) / (2.0 * h))
        curvature = (obj_plus - 2.0 * obj + obj_minus) / (h * h)

        return {
            "status": "finite_optimum",
            "policy": opt_policy,
            "objective": obj,
            "first_order_residual": first_order_residual,
            "curvature": curvature,
        }

    tail_sequence = (8.0, 16.0, 24.0, 32.0)
    tail_values = tuple(
        constrained_rlhf_objective(
            policy_from_delta(d), reference, reward_gap, beta, gamma
        )
        for d in tail_sequence
    )

    return {
        "status": "unbounded",
        "approached_boundary": "preferred",
        "increasing_tail_values": tail_values,
        "analytic_reason": "positive_gamma_objective_unbounded_at_preferred_boundary",
    }


def cpo_optimal_policy_margin(policy: TwoResponsePolicy, gamma: float) -> float:
    gamma = _finite("gamma", gamma)
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    return gamma * (1.0 / policy.preferred + 1.0 / policy.dispreferred)


def cpo_reference_margin(reference: TwoResponsePolicy, gamma: float) -> float:
    gamma = _finite("gamma", gamma)
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    return gamma * (1.0 / reference.preferred + 1.0 / reference.dispreferred)


def cpo_loss(delta: float, delta_ref: float, beta: float, margin: float) -> float:
    delta, delta_ref, beta = _loss_inputs(delta, delta_ref, beta)
    margin = _finite("margin", margin)
    return softplus(-beta * (delta - delta_ref) + margin)


def cpo_loss_derivative(
    delta: float, delta_ref: float, beta: float, margin: float
) -> float:
    delta, delta_ref, beta = _loss_inputs(delta, delta_ref, beta)
    margin = _finite("margin", margin)
    return -beta * sigmoid(-beta * (delta - delta_ref) + margin)


def scaled_dpo_soft_margin(delta: float, delta_ref: float, beta: float) -> float:
    delta, delta_ref, beta = _loss_inputs(delta, delta_ref, beta)
    return softplus(-beta * (delta - delta_ref)) / beta


def equations_13_16_residual(
    policy: TwoResponsePolicy,
    reference: TwoResponsePolicy,
    reward_gap: float,
    beta: float,
    margin: float,
) -> float:
    return abs(beta * (policy.delta - reference.delta) - (reward_gap - margin))
