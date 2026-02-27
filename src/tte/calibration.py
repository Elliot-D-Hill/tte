import torch
from torch import Tensor

from tte.utils import _normalize_risk_shape


def _grouped_observed_risk_at_time(
    bin_ids_time_sorted: Tensor,  # (N,)
    time_bucket: Tensor,  # (N,)
    event_indicator_sorted: Tensor,  # (N,) in {0,1}
    ones_subject: Tensor,  # (N,)
    n_bins: int,
    n_unique_times: int,
    eval_index: int,
) -> Tensor:
    """
    Unweighted per-bin KM observed risk (1 - S_hat) at one evaluation index.
    """
    dtype = event_indicator_sorted.dtype
    device = event_indicator_sorted.device
    if eval_index < 0 or n_unique_times == 0:
        return torch.zeros(n_bins, device=device, dtype=dtype)
    linear_index = bin_ids_time_sorted * n_unique_times + time_bucket
    cell_count = n_bins * n_unique_times
    counts_flat = torch.zeros(cell_count, device=device, dtype=dtype)
    counts_flat.scatter_add_(0, linear_index, ones_subject)
    events_flat = torch.zeros(cell_count, device=device, dtype=dtype)
    events_flat.scatter_add_(0, linear_index, event_indicator_sorted)
    counts = counts_flat.view(n_bins, n_unique_times)
    events_at_time = events_flat.view(n_bins, n_unique_times)
    at_risk = torch.flip(torch.cumsum(torch.flip(counts, dims=[1]), dim=1), dims=[1])
    hazard = events_at_time / torch.clamp(at_risk, min=1e-8)
    survival = torch.cumprod(1.0 - hazard, dim=1)
    return 1.0 - survival[:, eval_index]


def _sorted_bin_assignments(
    risk: Tensor,
    weights: Tensor,
    n_bins: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Sort by risk per time and assign each sorted rank to a weighted-frequency bin.
    Returns:
      order         : (N,T)
      risk_sorted   : (N,T)
      weights_sorted: (N,T)
      total_weight  : (T,)
      bin_ids_sorted: (N,T) values in [0, B-1]
    """
    device = risk.device
    dtype = risk.dtype
    n_samples, n_times = risk.shape
    order = torch.argsort(risk, dim=0, descending=True)
    risk_sorted = torch.gather(risk, 0, order)
    weights_sorted = weights[order]
    total_weight = weights_sorted.sum(dim=0)
    cumulative_weight = torch.cumsum(weights_sorted, dim=0)
    bin_steps = torch.arange(1, n_bins + 1, device=device, dtype=dtype).unsqueeze(0)
    thresholds = (total_weight / n_bins).unsqueeze(1) * bin_steps
    bin_ends = torch.searchsorted(
        cumulative_weight.transpose(0, 1).contiguous(),
        thresholds,
        right=False,
    )
    bin_ends = torch.cummax(bin_ends.clamp(max=n_samples), dim=1).values
    bin_ends[:, -1] = n_samples
    rank_positions = (
        torch.arange(n_samples, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(n_times, n_samples)
        .contiguous()
    )
    bin_ids_by_time = torch.searchsorted(
        bin_ends[:, :-1].contiguous(),
        rank_positions,
        right=True,
    )
    bin_ids_sorted = bin_ids_by_time.transpose(0, 1).contiguous()
    return order, risk_sorted, weights_sorted, total_weight, bin_ids_sorted


def _bin_aggregates_from_sorted(
    risk_sorted: Tensor,
    weights_sorted: Tensor,
    bin_ids_sorted: Tensor,
    n_bins: int,
) -> tuple[Tensor, Tensor]:
    """
    Shared per-(time,bin) aggregates:
      - weighted_risk_sum
      - bin_weight_sum
    """
    n_samples, n_times = risk_sorted.shape
    device = risk_sorted.device
    dtype = risk_sorted.dtype
    time_offsets = (
        torch.arange(n_times, device=device, dtype=torch.long) * n_bins
    ).unsqueeze(0)
    flat_index = (bin_ids_sorted + time_offsets).reshape(-1)
    flat_weights = weights_sorted.reshape(-1)
    flat_weighted_risk = (risk_sorted * weights_sorted).reshape(-1)
    num_cells = n_times * n_bins
    bin_weight_sum = torch.zeros(num_cells, device=device, dtype=dtype)
    weighted_risk_sum = torch.zeros(num_cells, device=device, dtype=dtype)
    bin_weight_sum.scatter_add_(0, flat_index, flat_weights)
    weighted_risk_sum.scatter_add_(0, flat_index, flat_weighted_risk)
    return weighted_risk_sum.view(n_times, n_bins), bin_weight_sum.view(n_times, n_bins)


def _expected_from_bin_aggregates(
    weighted_risk_sum: Tensor,
    bin_weight_sum: Tensor,
    eps: float,
) -> Tensor:
    """Per-(time,bin) weighted mean predicted risk."""
    return weighted_risk_sum / torch.clamp(bin_weight_sum, min=eps)


def _bin_weights_from_bin_aggregates(
    bin_weight_sum: Tensor,
    total_weight: Tensor,
    eps: float,
) -> Tensor:
    """Per-(time,bin) normalized bin mass."""
    return bin_weight_sum / torch.clamp(total_weight.unsqueeze(1), min=eps)


def _observed_from_grouped_km(
    order: Tensor,
    bin_ids_sorted: Tensor,
    event: Tensor,
    event_time: Tensor,
    eval_time: Tensor,
    n_bins: int,
    dtype: torch.dtype,
) -> Tensor:
    """
    Compute per-(time,bin) observed event risk via grouped unweighted KM.
    """
    n_samples, n_times = order.shape
    device = event_time.device
    time_order = torch.argsort(event_time)
    event_sorted = event[time_order]
    event_indicator_sorted = (event_sorted == 1).to(dtype=dtype)
    unique_times, time_bucket = torch.unique(
        event_time[time_order],
        sorted=True,
        return_inverse=True,
    )
    eval_indices = torch.searchsorted(unique_times, eval_time, right=True) - 1
    n_unique_times = unique_times.numel()
    subject_bin_ids = torch.empty((n_samples, n_times), device=device, dtype=torch.long)
    subject_bin_ids.scatter_(0, order, bin_ids_sorted)
    subject_bin_ids_time_sorted = subject_bin_ids[time_order]
    observed = torch.empty((n_times, n_bins), device=device, dtype=dtype)
    ones_subject = torch.ones(n_samples, device=device, dtype=dtype)
    for time_index in range(n_times):
        observed[time_index] = _grouped_observed_risk_at_time(
            bin_ids_time_sorted=subject_bin_ids_time_sorted[:, time_index],
            time_bucket=time_bucket,
            event_indicator_sorted=event_indicator_sorted,
            ones_subject=ones_subject,
            n_bins=n_bins,
            n_unique_times=n_unique_times,
            eval_index=int(eval_indices[time_index].item()),
        )
    return observed


def expected_observed_timebins(
    risk: Tensor,  # (N,T) CDF F(t)=P(T≤t)
    event: Tensor,  # (N,)
    event_time: Tensor,  # (N,)
    eval_time: Tensor,  # (T,)
    n_bins: int,
    weights: Tensor | None = None,  # (N,) IPCW subject weights
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Core routine used by both 1-calibration and ICI (IPCW-aware).

    For each t_j:
      - sort subjects by F_i(t_j) descending
      - split into ~equal *weighted* frequency bins (by cumulative sum of `weights`)
      - expected[j,b] = weighted mean(F_i(t_j)) in bin b
      - observed[j,b] = 1 - KM_hat(t_j) computed within that bin (unweighted KM)
      - bin_weights[j,b] = (sum of subject weights in bin b) / (sum of weights over all subjects)

    Returns:
      expected    : (T,B)
      observed    : (T,B)
      bin_weights : (T,B) rows sum to 1 (or 0 if N==0)
    """
    device = risk.device
    dtype = risk.dtype
    n_samples, n_times = risk.shape
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if weights is None:
        weights = torch.ones(n_samples, device=device, dtype=dtype)
    else:
        weights = weights.to(device=device, dtype=dtype)
    event = event.to(device=device)
    event_time = event_time.to(device=device)
    eval_time = eval_time.to(device=device)
    eps = torch.finfo(dtype).eps
    order, risk_sorted, weights_sorted, total_weight, bin_ids_sorted = (
        _sorted_bin_assignments(risk=risk, weights=weights, n_bins=n_bins)
    )
    weighted_risk_sum, bin_weight_sum = _bin_aggregates_from_sorted(
        risk_sorted=risk_sorted,
        weights_sorted=weights_sorted,
        bin_ids_sorted=bin_ids_sorted,
        n_bins=n_bins,
    )
    expected = _expected_from_bin_aggregates(
        weighted_risk_sum=weighted_risk_sum,
        bin_weight_sum=bin_weight_sum,
        eps=eps,
    )
    bin_weights = _bin_weights_from_bin_aggregates(
        bin_weight_sum=bin_weight_sum,
        total_weight=total_weight,
        eps=eps,
    )
    observed = _observed_from_grouped_km(
        order=order,
        bin_ids_sorted=bin_ids_sorted,
        event=event,
        event_time=event_time,
        eval_time=eval_time,
        n_bins=n_bins,
        dtype=dtype,
    )
    return expected, observed, bin_weights


def integrated_calibration_index(
    risk: Tensor,  # (N,T) or (N,) CDF F(t)=P(T≤t)
    event: Tensor,  # (N,) 1=event, 0=censored
    event_time: Tensor,  # (N,)
    eval_time: Tensor,  # (T,)
    n_bins: int = 10,
    integrate: bool = False,  # average over eval_time
    weights: Tensor | None = None,  # (N,) IPCW subject-level weights
) -> Tensor:
    """
    ECE-style Integrated Calibration Index (ICI).
    - KM-estimated observed risk per bin and time.
    - Uses subject-level IPCW weights if provided.
    Returns:
        (T,) ici_t unless integrate=True (scalar).
    """
    N, T = event.shape[0], eval_time.shape[0]
    risk = _normalize_risk_shape(risk, N, T)  # ensure (N,T)
    device, dtype = risk.device, risk.dtype
    if weights is None:
        weights = torch.ones(N, dtype=dtype, device=device)
    else:
        weights = weights.to(device=device, dtype=dtype)
    expected, observed, bin_weights = expected_observed_timebins(
        risk, event, event_time, eval_time, n_bins, weights=weights
    )
    ici_t = (bin_weights * (observed - expected).abs()).sum(dim=1)  # (T,)
    if integrate:
        return torch.trapz(ici_t, eval_time) / (eval_time[-1] - eval_time[0])
    return ici_t


def one_calibration(
    risk: Tensor,
    event: Tensor,
    event_time: Tensor,
    eval_time: Tensor,
    weights: Tensor | None = None,
    n_bins: int = 10,
) -> tuple[Tensor, Tensor, Tensor]:
    """1-calibration (Haider et al. or Greenwood–Nam–D'Agostino) using KM-observed risk per bin.
    Returns (chi2_t, observed, expected).
    """
    N, T = event.shape[0], eval_time.shape[0]
    risk = _normalize_risk_shape(risk, N, T)
    expected, observed, bin_weights = expected_observed_timebins(
        risk, event, event_time, eval_time, n_bins, weights=weights
    )
    if weights is None:
        total_weight = torch.as_tensor(float(N), device=risk.device, dtype=risk.dtype)
    else:
        total_weight = weights.to(device=risk.device, dtype=risk.dtype).sum()
    n_per_bin = bin_weights * total_weight  # (T,B)
    diff = observed - expected  # (T,B)
    denom = expected * (1.0 - expected) + 1e-12
    chi2 = (n_per_bin * (diff * diff) / denom).sum(dim=1)  # (T,)
    return chi2, observed, expected


def time_dependent_brier_score(
    risk: Tensor,  # (N,T) or (N,) CDF F(t)=P(T≤t)
    event: Tensor,  # (N,) 1=event, 0=censored
    event_time: Tensor,  # (N,)
    eval_time: Tensor,  # (T,)
    weights: Tensor | None = None,  # (N,) IPCW at T_i, defaults to 1s
    weights_eval_time: Tensor | None = None,  # (T,) IPCW at t_j, defaults to 1s
    integrate: bool = False,
) -> Tensor:
    N = event.shape[0]
    T = eval_time.shape[0]
    if risk.ndim == 1:
        risk = risk.unsqueeze(1).expand(N, T)
    elif risk.shape != (N, T):
        raise ValueError("risk must be shape (N,) or (N,T) matching eval_time.")
    device, dtype = risk.device, risk.dtype
    event = event.to(device)
    event_time = event_time.to(device)
    eval_time = eval_time.to(device)
    if weights is None:
        weights = torch.ones(N, dtype=dtype, device=device)
    if weights_eval_time is None:
        weights_eval_time = torch.ones(T, dtype=dtype, device=device)
    weight_i = weights.unsqueeze(1)  # (N,1)
    weight_j = weights_eval_time.unsqueeze(0)  # (1,T)
    survival = 1.0 - risk  # (N,T)
    event_col = event_time.unsqueeze(1)  # (N,1)
    time_row = eval_time.unsqueeze(0)  # (1,T)
    is_case = (event.bool().unsqueeze(1) & (event_col <= time_row)).to(dtype)  # (N,T)
    is_control = (event_col > time_row).to(dtype)  # (N,T)
    # residual = S^2 * 1{case} * w_i   +  (1-S)^2 * 1{control} * w_j
    residual = (survival.square() * is_case * weight_i) + (
        (1 - survival).square() * is_control * weight_j
    )
    brier_score_t = residual.mean(dim=0)  # (T,), divides by N exactly
    if integrate:
        return torch.trapz(brier_score_t, eval_time) / (eval_time[-1] - eval_time[0])
    return brier_score_t
