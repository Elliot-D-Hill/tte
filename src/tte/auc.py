import torch
from torch import Tensor

from tte.km import kaplan_meier
from tte.utils import _normalize_risk_shape, make_ranked_masks


def _integrate_over_survival(
    score: Tensor, event: Tensor, event_time: Tensor, eval_time: Tensor
) -> Tensor:
    S = kaplan_meier(event, event_time, eval_time)  # (T,)
    prev = torch.cat([score.new_tensor([1.0]), S[:-1]])
    dS = prev - S
    mask = torch.isfinite(score) & (dS > 0)
    num = (score[mask] * dS[mask]).sum()
    denom = dS.sum()
    return num / denom if denom > 0 else score.new_tensor(float("nan"))


def _partial_pairs(
    cases: Tensor, cumulative_control: Tensor, n_control: Tensor, alpha: float
) -> Tensor:
    """
    Σ_i cases[i,·] * max(α * n_control[·] - cumulative_control[i,·], 0) → (T,)
    """
    cap = (alpha * n_control).unsqueeze(0)  # (1,T)
    selected_after = torch.clamp(cap - cumulative_control, min=0.0)
    return (cases * selected_after).sum(dim=0)


def time_dependent_auc(
    risk: Tensor,
    event: Tensor,
    event_time: Tensor,
    eval_time: Tensor,
    weights: Tensor | None = None,
    integrate: bool = False,
) -> Tensor:
    """
    Cumulative/dynamic AUC_t. If fpr_range is None: full AUC_t; else partial AUC_t over FPR ∈ [α0, α1].
    Returns (T,) unless integrate=True → scalar.
    """
    N, T = event.shape[0], eval_time.shape[0]
    risk = _normalize_risk_shape(risk, N, T)
    cases, controls, _, n_case, n_control = make_ranked_masks(
        risk, event, event_time, eval_time, weights
    )
    cumulative_control = torch.cumsum(controls, dim=0)
    total_pairs = torch.as_tensor(n_case * n_control)
    controls_after = n_control.unsqueeze(0) - cumulative_control
    concordant_pairs = (cases * controls_after).sum(dim=0)
    auc_t = torch.full_like(total_pairs, torch.nan, dtype=risk.dtype)
    mask = total_pairs > 0
    auc_t[mask] = concordant_pairs[mask] / total_pairs[mask]
    return (
        _integrate_over_survival(auc_t, event, event_time, eval_time)
        if integrate
        else auc_t
    )


def time_dependent_pauc(
    risk: Tensor,
    event: Tensor,
    event_time: Tensor,
    eval_time: Tensor,
    fpr_range: tuple[float, float],
    weights: Tensor | None = None,
    integrate: bool = False,
    scale: bool = False,
) -> Tensor:
    """
    Cumulative/dynamic partial AUC_t over FPR ∈ [α0, α1].
    Returns (T,) unless integrate=True → scalar.
    """
    N, T = event.shape[0], eval_time.shape[0]
    risk = _normalize_risk_shape(risk, N, T)
    cases, controls, _, n_case, n_control = make_ranked_masks(
        risk, event, event_time, eval_time, weights
    )
    cumulative_control = torch.cumsum(controls, dim=0)
    total_pairs = torch.as_tensor(n_case * n_control)
    a0, a1 = fpr_range
    if not (0.0 <= a0 < a1 <= 1.0):
        raise ValueError("fpr_range must satisfy 0 ≤ α0 < α1 ≤ 1.")
    a1_pairs = _partial_pairs(cases, cumulative_control, n_control, a1)
    a0_pairs = _partial_pairs(cases, cumulative_control, n_control, a0)
    num = a1_pairs - a0_pairs
    pauc_t = torch.full_like(total_pairs, torch.nan, dtype=risk.dtype)
    mask = total_pairs > 0
    pauc_t[mask] = num[mask] / total_pairs[mask]
    if scale:
        width = a1 - a0
        if width <= 0:
            raise ValueError("invalid fpr_range width")
        pauc_t = pauc_t / width
    return (
        _integrate_over_survival(pauc_t, event, event_time, eval_time)
        if integrate
        else pauc_t
    )
