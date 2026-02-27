import torch
import torch.nn.functional as F
from torch import Tensor

from tte.km import kaplan_meier


def _normalize_risk_shape(risk: Tensor, n: int, t: int, name: str = "risk") -> Tensor:
    """Return risk as (N,T); accept (N,) or (N,T)."""
    if risk.ndim == 1:
        return risk.unsqueeze(1).expand(n, t)
    if risk.shape == (n, t):
        return risk
    raise ValueError(f"{name} must be (N,) or (N,T)")


def _make_raw_masks(
    event: Tensor, event_time: Tensor, eval_time: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Return:
        is_case    : (N,T) 1 if event by t_j
        is_control : (N,T) 1 if survived/censored beyond t_j
    """
    device, dtype = event.device, event.dtype
    event_col = event_time.to(device).unsqueeze(1)  # (N,1)
    time_row = eval_time.to(device).unsqueeze(0)  # (1,T)
    is_case = (event.bool().unsqueeze(1) & (event_col <= time_row)).to(dtype)
    is_control = (event_col > time_row).to(dtype)
    return is_case, is_control


def make_ranked_masks(
    risk: Tensor,
    event: Tensor,
    event_time: Tensor,
    eval_time: Tensor,
    weights: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Return (sorted by decreasing risk):
        cases     : (N,T)
        controls  : (N,T)
        valid     : (N,T)  cases + controls
        n_case    : (T,)
        n_control : (T,)
    """
    N, T = event.shape[0], eval_time.shape[0]
    risk = _normalize_risk_shape(risk, N, T)
    is_case, is_control = _make_raw_masks(event, event_time, eval_time)
    w_case = (
        torch.ones(N, dtype=risk.dtype, device=risk.device)
        if weights is None
        else weights.to(dtype=risk.dtype, device=risk.device)
    )
    weighted_case = is_case * w_case.unsqueeze(1)
    controls = is_control
    valid = weighted_case + controls
    order = torch.argsort(risk, dim=0, descending=True)
    cases = torch.gather(weighted_case, 0, order)
    controls = torch.gather(controls, 0, order)
    valid = torch.gather(valid, 0, order)
    n_case = cases.sum(0)
    n_control = controls.sum(0)
    return cases, controls, valid, n_case, n_control


def ipcw(
    event: Tensor,  # (N,) 1=event, 0=censor
    event_time: Tensor,  # (N,)
    eval_time: Tensor,  # (T,)
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """
    Returns:
      w_i : (N,)  subject-level 1 / Ĝ(T_i-)
      w_t : (T,)  time-level    1 / Ĝ(t-)
    """
    device = event_time.device
    censor_indicator = 1 - event  # censoring indicator (1=censor)
    # KM of censoring distribution; t_grid ascending, len M
    time_grid, censor_survival = kaplan_meier(censor_indicator, event_time)
    # Left-continuous step function: use value strictly before time (T_i-) to avoid bias at jumps
    idx_i = torch.searchsorted(time_grid, event_time, right=False) - 1  # (N,)
    idx_t = torch.searchsorted(time_grid, eval_time, right=False) - 1  # (T,)
    # Clamp to [-1, M-1]; for -1 (before first jump), use G(0-)=1.0
    idx_i = idx_i.clamp(min=-1, max=censor_survival.numel() - 1)
    idx_t = idx_t.clamp(min=-1, max=censor_survival.numel() - 1)
    # Gather with padding of G(0-)=1.0; length M+1, G_pad0[0]=1.0
    censor_survival_pad = torch.cat(
        [censor_survival.new_tensor([1.0]), censor_survival]
    )
    weight_i = 1.0 / torch.clamp(censor_survival_pad[idx_i + 1], min=eps)
    weight_t = 1.0 / torch.clamp(censor_survival_pad[idx_t + 1], min=eps)
    return weight_i.to(device), weight_t.to(device)


def cumulative_incidence(
    event: Tensor,  # (N,)
    event_time: Tensor,  # (N,)
    eval_time: Tensor,  # (T,)
) -> Tensor:
    """
    (T,) cumulative incidence at each t_j.
    Single-event setting: F(t) = 1 - S_KM(t).
    """
    survival = kaplan_meier(event, event_time, eval_time)  # (T,)
    return 1.0 - survival


def cumulative_positive_count(
    event: Tensor,  # (N,)
    event_time: Tensor,  # (N,)
    eval_time: Tensor,  # (T,)
    integrate: bool = False,
) -> Tensor:
    """
    Returns: (T,) positives at each t_j
    """
    cases, _ = _make_raw_masks(event, event_time, eval_time)
    n_cases = cases.sum(dim=0)
    if integrate:
        return n_cases.sum()
    return n_cases


def next_event_times(tokens: Tensor, time: Tensor, vocab_size: int) -> Tensor:
    time = time.squeeze(-1) if time.dim() == 3 else time
    event_mask = F.one_hot(tokens, num_classes=vocab_size).to(torch.bool)
    event_times = time.unsqueeze(-1).expand(-1, -1, vocab_size)
    event_times = event_times.masked_fill(~event_mask, torch.inf)
    # Suffix min, then shift forward by 1 to get strictly next occurrence
    event_times_reversed = torch.flip(event_times, dims=[1])
    suffix_min = torch.flip(torch.cummin(event_times_reversed, dim=1).values, dims=[1])
    next_time = F.pad(suffix_min[:, 1:], (0, 0, 0, 1), value=torch.inf)
    time_to_event = next_time - time.unsqueeze(-1)
    return time_to_event
