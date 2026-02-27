import torch

from tte.calibration import expected_observed_timebins, one_calibration
from tte.km import kaplan_meier


def _reference_expected_observed_timebins(
    risk: torch.Tensor,
    event: torch.Tensor,
    event_time: torch.Tensor,
    eval_time: torch.Tensor,
    n_bins: int,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = risk.device
    dtype = risk.dtype
    n_samples, n_times = risk.shape
    if weights is None:
        weights = torch.ones(n_samples, device=device, dtype=dtype)
    else:
        weights = weights.to(device=device, dtype=dtype)
    event = event.to(device=device)
    event_time = event_time.to(device=device)
    eval_time = eval_time.to(device=device)
    eps = torch.finfo(dtype).eps

    expected_per_time = []
    observed_per_time = []
    bin_weights_per_time = []
    for time_index in range(n_times):
        predicted_risk_at_time = risk[:, time_index]
        sorted_indices = torch.argsort(predicted_risk_at_time, dim=0, descending=True)
        risk_sorted = predicted_risk_at_time[sorted_indices]
        weights_sorted = weights[sorted_indices]
        event_time_sorted = event_time[sorted_indices]
        event_sorted = event[sorted_indices]

        total_weight = weights_sorted.sum()
        target_bin_weight = total_weight / n_bins
        cumulative_weight = torch.cumsum(weights_sorted, dim=0)
        bin_starts = []
        bin_ends = []
        prev_end = 0
        for bin_index in range(n_bins):
            threshold = (bin_index + 1) * target_bin_weight
            end_index = torch.searchsorted(
                cumulative_weight,
                torch.as_tensor(threshold, device=device, dtype=dtype),
            ).item()
            end_index = max(end_index, prev_end)
            end_index = min(end_index, n_samples)
            bin_starts.append(prev_end)
            bin_ends.append(end_index)
            prev_end = end_index
        bin_ends[-1] = n_samples

        expected_bins = []
        observed_bins = []
        bin_weights_for_time = []
        eval_time_at_j = eval_time[time_index]
        for start, end in zip(bin_starts, bin_ends):
            slice_i = slice(start, end)
            weight_sum_bin = weights_sorted[slice_i].sum()
            bin_weights_for_time.append(weight_sum_bin)
            weighted_expected = (risk_sorted[slice_i] * weights_sorted[slice_i]).sum()
            expected_bins.append(
                weighted_expected / torch.clamp(weight_sum_bin, min=eps)
            )
            survival_at_time = kaplan_meier(
                event=event_sorted[slice_i],
                event_time=event_time_sorted[slice_i],
                eval_time=eval_time_at_j.unsqueeze(0)
                if eval_time_at_j.ndim == 0
                else eval_time_at_j,
            ).squeeze()
            observed_bins.append(1.0 - survival_at_time)

        expected_per_time.append(torch.stack(expected_bins))
        observed_per_time.append(torch.stack(observed_bins))
        bin_weights_for_time = torch.stack(bin_weights_for_time) / torch.clamp(
            total_weight, min=eps
        )
        bin_weights_per_time.append(bin_weights_for_time)

    expected = torch.stack(expected_per_time, dim=0)
    observed = torch.stack(observed_per_time, dim=0)
    bin_weights = torch.stack(bin_weights_per_time, dim=0)
    return expected, observed, bin_weights


def test_expected_observed_timebins_matches_reference():
    torch.manual_seed(0)
    n_samples, n_times, n_bins = 97, 11, 7
    risk = torch.rand(n_samples, n_times, dtype=torch.float64)
    event = (torch.rand(n_samples) < 0.35).to(torch.int64)
    event_time = torch.randint(1, 50, (n_samples,), dtype=torch.int64).to(torch.float64)
    eval_time = torch.linspace(0, 60, n_times, dtype=torch.float64)
    weights = 0.1 + 2.0 * torch.rand(n_samples, dtype=torch.float64)

    expected_ref, observed_ref, bin_weights_ref = _reference_expected_observed_timebins(
        risk, event, event_time, eval_time, n_bins=n_bins, weights=weights
    )
    expected, observed, bin_weights = expected_observed_timebins(
        risk, event, event_time, eval_time, n_bins=n_bins, weights=weights
    )

    assert torch.allclose(
        observed.to(dtype=observed_ref.dtype),
        observed_ref,
        rtol=1e-7,
        atol=1e-7,
    )
    assert torch.allclose(expected, expected_ref, rtol=1e-6, atol=1e-6)
    assert torch.allclose(bin_weights, bin_weights_ref, rtol=1e-6, atol=1e-6)


def test_expected_observed_timebins_handles_empty_bins():
    risk = torch.tensor(
        [
            [0.99, 0.90, 0.80],
            [0.70, 0.60, 0.50],
            [0.50, 0.40, 0.30],
            [0.40, 0.30, 0.20],
            [0.30, 0.20, 0.10],
            [0.20, 0.10, 0.05],
        ],
        dtype=torch.float64,
    )
    event = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.int64)
    event_time = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float64)
    eval_time = torch.tensor([0.0, 3.0, 10.0], dtype=torch.float64)
    weights = torch.tensor([1000.0, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9], dtype=torch.float64)

    expected, observed, bin_weights = expected_observed_timebins(
        risk, event, event_time, eval_time, n_bins=5, weights=weights
    )
    empty_bins = bin_weights == 0
    assert empty_bins.any()
    assert torch.isfinite(expected).all()
    assert torch.isfinite(observed).all()
    assert torch.all(expected[empty_bins] == 0)
    assert torch.all(observed[empty_bins] == 0)


def test_expected_observed_timebins_all_censored_observed_zero():
    torch.manual_seed(1)
    n_samples, n_times, n_bins = 30, 8, 6
    risk = torch.rand(n_samples, n_times, dtype=torch.float64)
    event = torch.zeros(n_samples, dtype=torch.int64)
    event_time = torch.randint(1, 20, (n_samples,), dtype=torch.int64).to(torch.float64)
    eval_time = torch.linspace(0, 25, n_times, dtype=torch.float64)

    _, observed, _ = expected_observed_timebins(
        risk, event, event_time, eval_time, n_bins=n_bins
    )
    assert torch.all(observed == 0)


def test_expected_observed_timebins_eval_time_extremes():
    risk = torch.tensor(
        [
            [0.7, 0.8, 0.9],
            [0.4, 0.3, 0.2],
            [0.6, 0.5, 0.4],
            [0.1, 0.2, 0.3],
            [0.9, 0.6, 0.1],
            [0.2, 0.9, 0.8],
            [0.8, 0.4, 0.5],
            [0.3, 0.7, 0.6],
        ],
        dtype=torch.float64,
    )
    event = torch.tensor([1, 0, 1, 0, 1, 1, 0, 0], dtype=torch.int64)
    event_time = torch.tensor([2, 3, 5, 7, 11, 13, 17, 19], dtype=torch.float64)
    eval_time = torch.tensor([0.5, 6.0, 30.0], dtype=torch.float64)
    weights = torch.tensor([1.2, 0.8, 1.1, 0.9, 1.0, 1.5, 1.3, 0.7], dtype=torch.float64)

    expected_ref, observed_ref, bin_weights_ref = _reference_expected_observed_timebins(
        risk, event, event_time, eval_time, n_bins=4, weights=weights
    )
    expected, observed, bin_weights = expected_observed_timebins(
        risk, event, event_time, eval_time, n_bins=4, weights=weights
    )

    assert torch.allclose(
        observed.to(dtype=observed_ref.dtype),
        observed_ref,
        rtol=1e-7,
        atol=1e-7,
    )
    assert torch.allclose(expected, expected_ref, rtol=1e-6, atol=1e-6)
    assert torch.allclose(bin_weights, bin_weights_ref, rtol=1e-6, atol=1e-6)
    assert torch.all(observed[0] == 0)


def test_one_calibration_returns_per_time_and_matches_manual():
    torch.manual_seed(2)
    n_samples, n_times, n_bins = 60, 9, 5
    risk = torch.rand(n_samples, n_times, dtype=torch.float64)
    event = (torch.rand(n_samples) < 0.4).to(torch.int64)
    event_time = torch.randint(1, 80, (n_samples,), dtype=torch.int64).to(torch.float64)
    eval_time = torch.linspace(0, 90, n_times, dtype=torch.float64)
    weights = 0.2 + torch.rand(n_samples, dtype=torch.float64)

    chi2, observed, expected = one_calibration(
        risk, event, event_time, eval_time, n_bins=n_bins, weights=weights
    )
    _, _, bin_weights = expected_observed_timebins(
        risk, event, event_time, eval_time, n_bins=n_bins, weights=weights
    )

    total_weight = weights.sum()
    n_per_bin = bin_weights * total_weight
    denom = expected * (1.0 - expected) + 1e-12
    chi2_manual = (n_per_bin * (observed - expected).square() / denom).sum(dim=1)

    assert chi2.shape == (n_times,)
    assert observed.shape == (n_times, n_bins)
    assert expected.shape == (n_times, n_bins)
    assert torch.allclose(chi2, chi2_manual, rtol=1e-10, atol=1e-10)
