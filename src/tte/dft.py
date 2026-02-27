import torch
import torch.nn as nn
from torch import Tensor


def _reduce_loss(nll: Tensor, mask: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        total = nll.sum()
        denom = mask.sum()
        return total / denom.clamp_min(1)
    raise ValueError(f"Unknown reduction: {reduction}")


def _validate_inputs(
    logits: Tensor,
    indicator: Tensor,
    event_time: Tensor,
    mask: Tensor | None = None,
) -> None:
    if logits.dim() == 3:
        if indicator.dim() != 2:
            raise ValueError("Expected indicator with shape (B, L).")
        if mask is not None and mask.dim() != 2:
            raise ValueError("Expected mask with shape (B, L).")
        if event_time.dim() != 3:
            raise ValueError("Expected event_time with shape (B, L, K).")
    elif logits.dim() == 4:
        if indicator.dim() != 3:
            raise ValueError("Expected indicator with shape (B, L, E).")
        if mask is not None and mask.dim() not in (2, 3):
            raise ValueError("Expected mask with shape (B, L) or (B, L, E).")
        if event_time.dim() != 4:
            raise ValueError("Expected event_time with shape (B, L, E, K).")
        if indicator.size(2) != logits.size(2):
            raise ValueError("Event dimension mismatch between logits and indicator.")
        if event_time.size(2) != logits.size(2):
            raise ValueError("Event dimension mismatch between logits and event_time.")
    else:
        raise ValueError("Expected logits with shape (B, L, C) or (B, L, E, C).")
    if event_time.size(-1) != logits.size(-1) - 1:
        raise ValueError("Expected event_time last dim to match logits.size(-1) - 1.")


class DiscreteFailureTimeNLL(nn.Module):
    """
    Negative log-likelihood loss for discrete-time survival analysis.
    """

    def __init__(
        self, ignore_index: int, reduction: str = "mean", eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.ignore_index: int = ignore_index
        self.eps: float = eps
        self.reduction: str = reduction

    def forward(
        self,
        logits: Tensor,
        indicator: Tensor,
        event_time: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            logits: Logits with shape (B, L, C) or (B, L, E, C).
            indicator: Event indicator with shape (B, L) or (B, L, E).
            event_time: Discrete event time encoding with shape (B, L, K) or
                (B, L, E, K).
            mask: Optional mask with shape (B, L) or (B, L, E).

        Returns:
            NLL loss with shape (B, L) or (B, L, E) when reduction="none",
            otherwise a scalar.
        """
        _validate_inputs(logits, indicator, event_time, mask)
        interval_pmf = torch.softmax(logits, dim=-1)[..., :-1]
        if mask is None:
            mask = indicator != self.ignore_index
        elif mask.dim() == 2 and indicator.dim() == 3:
            mask = mask.unsqueeze(-1)
        mask = mask.to(logits.dtype)
        indicator = indicator.to(logits.dtype)
        event = indicator * mask
        censor = (1.0 - indicator) * mask
        in_interval = (event_time > 0).to(logits.dtype)
        event_likelihood = (in_interval * interval_pmf).sum(dim=-1)
        log_likelihood = event_likelihood.clamp_min(self.eps).log()
        cumulative_event_prob = (event_time * interval_pmf).sum(dim=-1)
        survival_probability = 1.0 - cumulative_event_prob
        log_probability = survival_probability.clamp_min(self.eps).log()
        nll = -(event * log_likelihood + censor * log_probability)
        return _reduce_loss(nll, mask, self.reduction)


def discretize_event_time(
    event_time: Tensor, indicator: Tensor, boundaries: Tensor
) -> Tensor:
    interval_start = boundaries[:-1].view(*([1] * event_time.dim()), -1)
    interval_end = boundaries[1:].view(*([1] * event_time.dim()), -1)
    interval_width = interval_end - interval_start
    event_time = event_time.unsqueeze(-1)
    exposure = ((event_time - interval_start) / interval_width).clamp(0, 1)
    in_interval = (event_time > interval_start) & (event_time <= interval_end)
    indicator = indicator.unsqueeze(-1).to(exposure.dtype)
    return indicator * (exposure * in_interval) + (1.0 - indicator) * exposure
