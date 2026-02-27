from typing import overload

import torch
from torch import Tensor


def _step_interp(source_x: Tensor, source_y: Tensor, target_x: Tensor) -> Tensor:
    indices = torch.searchsorted(source_x, target_x, right=True) - 1
    indices = indices.clamp(min=0)
    return source_y[indices]


@overload
def kaplan_meier(
    event: Tensor, event_time: Tensor, eval_time: None = None
) -> tuple[Tensor, Tensor]: ...


@overload
def kaplan_meier(event: Tensor, event_time: Tensor, eval_time: Tensor) -> Tensor: ...


def kaplan_meier(
    event: Tensor, event_time: Tensor, eval_time: Tensor | None = None
) -> Tensor | tuple[Tensor, Tensor]:
    num_subjects = event_time.shape[0]
    device = event_time.device
    sorted_indices = torch.argsort(event_time)
    times_sorted = event_time[sorted_indices]
    events_sorted = event[sorted_indices]
    unique_event_times, events_at_time = torch.unique(
        times_sorted[events_sorted == 1], return_counts=True
    )
    at_risk_counts = num_subjects - torch.searchsorted(times_sorted, unique_event_times)
    survival = torch.cumprod(
        1.0 - events_at_time / torch.clamp(at_risk_counts, min=1e-8), dim=0
    )
    zeros = torch.zeros(1, device=device, dtype=event_time.dtype)
    time_grid = torch.cat([zeros, unique_event_times])
    ones = torch.ones(1, device=device, dtype=survival.dtype)
    survival_grid = torch.cat([ones, survival])
    if eval_time is None:
        return time_grid, survival_grid
    return _step_interp(time_grid, survival_grid, eval_time)
