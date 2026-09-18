from umirage.protected import (
    PropagationWeights,
    SourceScores,
    no_propagation_risk,
    propagated_risk,
    selectivity,
    source_action,
)


def test_exact_same_signal_risks_are_bounded():
    scores = SourceScores(.2, .7, .4)
    control = no_propagation_risk(scores)
    prop = propagated_risk(scores, PropagationWeights(1, 1, 1, 1, 1))
    assert 0 <= control <= 1
    assert 0 <= prop <= 1


def test_propagated_risk_is_monotonic_in_each_source():
    weights = PropagationWeights(1, 1, 1, 1, 1)
    base = SourceScores(.2, .3, .4)
    risk = propagated_risk(base, weights)
    assert propagated_risk(SourceScores(.5, .3, .4), weights) >= risk
    assert propagated_risk(SourceScores(.2, .6, .4), weights) >= risk
    assert propagated_risk(SourceScores(.2, .3, .7), weights) >= risk


def test_selectivity_is_positive_when_target_moves_most():
    control = SourceScores(.1, .1, .1)
    intervention = SourceScores(.2, .8, .2)
    sel, delta = selectivity(control, intervention, "integration")
    assert sel > 0
    assert delta["integration"] > delta["visual"]
    assert source_action("integration") == "compare_images"
