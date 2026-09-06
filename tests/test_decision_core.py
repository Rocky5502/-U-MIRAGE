import math

import pytest

from umirage.intervention import PolicyConfig, recommend_action
from umirage.propagation import PropagationWeights, SourceAwarePropagator


def test_propagator_is_bounded_and_posterior_normalized():
    risk, posterior = SourceAwarePropagator().propagate(0.25, 0.75, 0.40)
    assert 0.0 <= risk <= 1.0
    assert set(posterior) == {"visual", "integration", "reasoning"}
    assert math.isclose(sum(posterior.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert max(posterior, key=posterior.get) == "integration"


def test_propagator_is_monotonic_per_source():
    propagator = SourceAwarePropagator()
    base, _ = propagator.propagate(0.20, 0.20, 0.20)
    visual, _ = propagator.propagate(0.60, 0.20, 0.20)
    integration, _ = propagator.propagate(0.20, 0.60, 0.20)
    reasoning, _ = propagator.propagate(0.20, 0.20, 0.60)
    assert visual >= base
    assert integration >= base
    assert reasoning >= base


def test_propagator_clamps_out_of_range_inputs():
    risk, posterior = SourceAwarePropagator().propagate(-4.0, 2.0, 0.5)
    assert 0.0 <= risk <= 1.0
    assert max(posterior, key=posterior.get) == "integration"


def test_invalid_propagation_weights_are_rejected():
    with pytest.raises(ValueError):
        PropagationWeights(visual=-0.1)
    with pytest.raises(ValueError):
        PropagationWeights(
            visual=0.0,
            integration=0.0,
            reasoning=0.0,
            interaction=0.0,
            dominant=0.0,
        )


def test_policy_answers_low_risk_and_abstains_high_risk():
    posterior = {"visual": 0.70, "integration": 0.20, "reasoning": 0.10}
    assert recommend_action(0.10, posterior) == "answer"
    assert recommend_action(0.95, posterior) == "abstain"


def test_policy_routes_intermediate_risk_by_source():
    assert recommend_action(
        0.60, {"visual": 0.70, "integration": 0.20, "reasoning": 0.10}
    ) == "reinspect_image"
    assert recommend_action(
        0.60, {"visual": 0.10, "integration": 0.75, "reasoning": 0.15}
    ) == "compare_images"
    assert recommend_action(
        0.60, {"visual": 0.10, "integration": 0.20, "reasoning": 0.70}
    ) == "verify_reasoning"


def test_policy_falls_back_when_source_posterior_is_diffuse():
    posterior = {"visual": 0.34, "integration": 0.33, "reasoning": 0.33}
    assert recommend_action(0.60, posterior) == "verify_reasoning"


def test_invalid_policy_thresholds_are_rejected():
    with pytest.raises(ValueError):
        PolicyConfig(answer_threshold=0.9, abstain_threshold=0.8)
