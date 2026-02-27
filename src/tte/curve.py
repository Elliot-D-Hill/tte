import torch
from torch import Tensor

from tte.utils import _normalize_risk_shape


def _sorted_true_false_positive_counts(
    risk: Tensor,
    event: Tensor,
    event_time: Tensor,
    eval_time: Tensor,
    weights: Tensor | None = None,
) -> tuple[Tensor, ...]:
    """
    Returns
      thresholds (N+1,T) ascending, last row = inf
      true_positive_exclusive (N+1,T) for risk > threshold
      false_positive_exclusive (N+1,T) for risk > threshold
      total_positives (T,)
      total_negatives (T,)
    """
    N, T = event.numel(), eval_time.numel()
    risk = _normalize_risk_shape(risk, N, T)
    order = torch.argsort(risk, dim=0)
    thresholds = torch.gather(risk, 0, order)
    event_time_col = event_time.unsqueeze(1)
    eval_time_row = eval_time.unsqueeze(0)
    is_case = (event.bool().unsqueeze(1)) & (event_time_col <= eval_time_row)
    is_control = event_time_col > eval_time_row
    weights_case = (
        torch.ones(N, dtype=risk.dtype, device=risk.device)
        if weights is None
        else weights.to(dtype=risk.dtype, device=risk.device)
    )
    pos_sorted = torch.gather(
        (is_case.to(risk.dtype) * weights_case.unsqueeze(1)), 0, order
    )
    neg_sorted = torch.gather(is_control.to(risk.dtype), 0, order)
    pos_cumsum = torch.cumsum(pos_sorted, dim=0)
    neg_cumsum = torch.cumsum(neg_sorted, dim=0)
    total_positives = pos_cumsum[-1]
    total_negatives = neg_cumsum[-1]
    true_positive_exclusive = total_positives - pos_cumsum
    false_positive_exclusive = total_negatives - neg_cumsum
    zeros = torch.zeros_like(total_positives)
    true_positive_exclusive = torch.vstack([true_positive_exclusive, zeros])
    false_positive_exclusive = torch.vstack([false_positive_exclusive, zeros])
    inf_row = torch.full((T,), float("inf"), dtype=risk.dtype, device=risk.device)
    thresholds = torch.vstack([thresholds, inf_row.unsqueeze(0)])
    return (
        thresholds,
        true_positive_exclusive,
        false_positive_exclusive,
        total_positives,
        total_negatives,
    )


def time_dependent_roc(
    risk: Tensor,
    event: Tensor,
    event_time: Tensor,
    eval_time: Tensor,
    weights: Tensor | None = None,
) -> tuple[Tensor, ...]:
    """
    Returns
      false_positive_rate (N+1,T)
      true_positive_rate  (N+1,T)
      thresholds          (N+1,T)
    """
    thresholds, tp_excl, fp_excl, total_positives, total_negatives = (
        _sorted_true_false_positive_counts(risk, event, event_time, eval_time, weights)
    )
    total_positives_row = total_positives.unsqueeze(0)
    total_negatives_row = total_negatives.unsqueeze(0)
    true_positive_rate = torch.where(
        total_positives_row > 0,
        tp_excl / total_positives_row,
        torch.zeros_like(tp_excl),
    )
    false_positive_rate = torch.where(
        total_negatives_row > 0,
        fp_excl / total_negatives_row,
        torch.zeros_like(fp_excl),
    )
    return false_positive_rate, true_positive_rate, thresholds


def time_dependent_pr(
    risk: Tensor,
    event: Tensor,
    event_time: Tensor,
    eval_time: Tensor,
    weights: Tensor | None = None,
) -> tuple[Tensor, ...]:
    """
    Returns
      recall      (N+1,T)
      precision   (N+1,T)
      thresholds  (N+1,T)
    """
    thresholds, tp_excl, fp_excl, total_positives, _ = (
        _sorted_true_false_positive_counts(risk, event, event_time, eval_time, weights)
    )
    total_positives_row = total_positives.unsqueeze(0)
    recall = torch.where(
        total_positives_row > 0,
        tp_excl / total_positives_row,
        torch.zeros_like(tp_excl),
    )
    predicted_positive = tp_excl + fp_excl
    precision = torch.where(
        predicted_positive > 0,
        tp_excl / predicted_positive,
        torch.ones_like(predicted_positive),
    )
    return recall, precision, thresholds
