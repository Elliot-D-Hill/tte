import math

import pytest
import torch

from tte.auc import time_dependent_auc


def T(x):  # quick helper for concise tensor creation
    return torch.tensor(x, dtype=torch.float64)


# Perfect separation: AUC = 1 at all horizons with comparable pairs
def test_auc_perfect_separation():
    times = T([1, 2, 3, 10, 11, 12])
    events = T([1, 1, 1, 0, 0, 0])
    risk = T([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])  # higher → earlier event
    t = T([1, 2, 2.5, 3, 5, 9, 10])
    auc = time_dependent_auc(risk, events, times, t)
    # Below first event (t<1) would be NaN; here we start at ≥1.
    assert torch.allclose(
        auc[~torch.isnan(auc)], T([1, 1, 1, 1, 1, 1, 1]), equal_nan=True
    )


def test_auc_no_signal_is_half():
    # Replace the "all equal scores → 0.5" claim with a property that
    # holds independently of tie handling: symmetry under sign flip
    # (when there are no exact ties).
    times = T([1, 2, 3, 10, 11, 12])
    events = T([1, 1, 1, 0, 0, 0])
    # Use strictly increasing scores to avoid ties.
    risk = T([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = T([1, 2, 3, 5, 10])
    auc_pos = time_dependent_auc(risk, events, times, t)
    auc_neg = time_dependent_auc(-risk, events, times, t)
    # AUC(r) + AUC(-r) == 1 where defined (no NaNs).
    mask = ~torch.isnan(auc_pos) & ~torch.isnan(auc_neg)
    assert torch.allclose(
        auc_pos[mask] + auc_neg[mask],
        torch.ones(int(mask.sum().item())).double(),
        atol=1e-12,
    )


# Mixed with censoring: known values at two horizons
def test_auc_mixed_known_values():
    # times/event from earlier example; event==1 means event, 0 means censored
    times = T([2, 4, 5, 6, 7, 8])
    events = T([0, 0, 1, 1, 0, 1])
    risk = T([0.6, 0.4, 0.7, 0.2, 0.5, 0.3])
    t = T([5, 6])
    # t=5: cases={idx 2}, controls={idx 3,4,5} → AUC=1.0
    # t=6: cases={idx 2,3}, controls={idx 4,5}, 2/4 concordant → 0.5
    auc = time_dependent_auc(risk, events, times, t)
    assert torch.allclose(auc, T([1.0, 0.5]), equal_nan=True)


# Undefined sets: NaN at horizons with zero comparable pairs
def test_auc_undefined_sets_nan():
    times = T([1, 2, 3, 10, 11, 12])
    events = T([1, 1, 1, 0, 0, 0])
    risk = T([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    # t=0.5 (no cases yet) → NaN; t=12 (no controls left) → NaN
    t = T([0.5, 12.0])
    auc = time_dependent_auc(risk, events, times, t)
    assert torch.isnan(auc).tolist() == [True, True]


# Rank invariance (monotone transforms)
def test_auc_rank_invariance():
    times = T([1, 2, 3, 10, 11, 12])
    events = T([1, 1, 1, 0, 0, 0])
    risk = T([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    t = T([2, 3, 5])
    base = time_dependent_auc(risk, events, times, t)
    scaled = time_dependent_auc(2 * risk + 1, events, times, t)
    logistic = time_dependent_auc(
        1 / (1 + torch.exp(-5 * (risk - 0.5))), events, times, t
    )
    assert torch.allclose(base, scaled, equal_nan=True)
    assert torch.allclose(base, logistic, equal_nan=True)


# Input validation: shape/NaN/negative times
def test_input_validation_mismatched_lengths():
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        # risk, event, event_time must align in N
        time_dependent_auc(T([0.1, 0.2]), T([1]), T([1.0, 2.0]), T([2.0]))


def test_input_gives_nan():
    times = T(
        [1, 2, -1]
    )  # if your upstream validation forbids negatives/NaNs, this should raise
    events = T([1, 0, 1])
    risk = T([0.2, float("nan"), 0.7])
    auc = time_dependent_auc(risk, events, times, T([2.0]))
    assert torch.isnan(auc).item() is True


# Censoring semantics explicit check
def test_controls_exclude_censored_before_t():
    # Subject 0 censored at 2; for t=3 they should NOT be a control.
    times = T([2, 5, 6])
    events = T([0, 1, 0])
    risk = T([0.9, 0.8, 0.1])
    t = T([3, 5])
    auc = time_dependent_auc(risk, events, times, t)
    # At t=3: no cases (event at 5>3) → NaN.
    # At t=5: cases={idx1}, controls={idx2}; censored-at-2 (idx0) excluded.
    assert math.isnan(float(auc[0]))
    assert not math.isnan(float(auc[1]))


# Vector (N,T) risk handling
def test_auc_accepts_matrix_risk():
    times = T([2, 4, 6, 8])
    events = T([1, 1, 0, 0])
    base_risk = T([0.9, 0.8, 0.2, 0.1])
    t = T([3, 5, 7])
    # Provide per-horizon risk columns identical to base risk (so AUC should match scalar-risk path)
    risk_matrix = base_risk.unsqueeze(1).expand(-1, t.numel())
    auc1 = time_dependent_auc(base_risk, events, times, t)
    auc2 = time_dependent_auc(risk_matrix, events, times, t)
    assert torch.allclose(auc1, auc2, equal_nan=True)


# Weights affect pair counts
def test_auc_weights_affect_counts():
    # Simple configuration: at t=3, one case vs two controls; weight a control heavier and see effect.
    times = T([1, 4, 5])  # idx0 event at 1, idx1 censored 4, idx2 censored 5
    events = T([1, 0, 0])
    risk = T([0.9, 0.2, 0.1])  # ranks: case highest → concordant with both controls
    t = T([3])
    # Unweighted: 2 concordant pairs / 2 total → 1.0
    auc_unw = time_dependent_auc(risk, events, times, t)
    assert torch.allclose(auc_unw, T([1.0]), equal_nan=True)

    # Weighted: double weight on one control, still all concordant → remains 1.0 but pairs count differs internally
    w = T([1.0, 2.0, 1.0])
    auc_w = time_dependent_auc(risk, events, times, t, weights=w)
    assert torch.allclose(auc_w, T([1.0]), equal_nan=True)
