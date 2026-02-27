import torch
from torch import Tensor

from tte.utils import make_ranked_masks


def time_dependent_ap(
    risk: Tensor,
    event: Tensor,
    event_time: Tensor,
    eval_time: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    """(T,) cumulative/dynamic average precision at each eval_time."""
    cases, _, valid, n_case, _ = make_ranked_masks(
        risk, event, event_time, eval_time, weights
    )
    cumulative_case = torch.cumsum(cases, dim=0)
    cumulative_valid = torch.cumsum(valid, dim=0).clamp_(min=1)
    precision = cumulative_case / cumulative_valid
    ap_num = (precision * cases).sum(dim=0)
    ap_t = torch.full_like(ap_num, torch.nan, dtype=risk.dtype)
    mask = n_case > 0
    ap_t[mask] = ap_num[mask] / n_case[mask]
    return ap_t
