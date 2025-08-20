from __future__ import annotations

from typing import (
    Iterable,
    Literal,
    Optional,
    Union,
)

from torch import LongTensor, Tensor

from ...processes import StationaryProcessFDD
from ..helpers import ACF, _lags_to_idx_tensor
from .base_loss import BaseStatLoss


class ACFLoss(BaseStatLoss):
    """Autocorrelation loss."""

    @classmethod
    def analytical(
        cls,
        distr: StationaryProcessFDD,
        lags: Optional[Union[int, Iterable[int], LongTensor]] = None,
        *,
        acf_method: Literal["fft", "brute"] = "fft",
        **configuration_opts,
    ) -> "ACFLoss":
        if lags is None:
            lags = distr.times.numel()

        lags_idx: LongTensor = (
            _lags_to_idx_tensor(lags, device=distr.process.device) if not isinstance(lags, LongTensor) else lags
        )

        target = distr.process.acf(distr.times[lags_idx])
        stat_fn = ACF(
            lags,
            method=acf_method,
            data_batch_first=True,
        )
        return cls(target, stat_fn, acf_method=acf_method, lags=lags, **configuration_opts)

    @classmethod
    def empirical(
        cls,
        data: Tensor,
        lags: Optional[Union[int, Iterable[int], LongTensor]] = None,
        *,
        acf_method: Literal["fft", "brute"] = "fft",
        **configuration_opts,
    ) -> "ACFLoss":
        """Build an :class:`ACFLoss` from a set of observed trajectories.

        Parameters
        ----------
        data : Tensor
            Empirical trajectories ``(B, D, T)``.
        lags : int
            Number of lags to include in the target.
        kwargs
            Passed to the underlying :class:`BaseStatLoss`.

        """
        if lags is None:
            lags = data.shape[0]
        if not isinstance(lags, LongTensor):
            lags = _lags_to_idx_tensor(lags, device=data.device)
        stat_fn = ACF(
            lags,
            method=acf_method,
            data_batch_first=True,
        )
        target = stat_fn(data)
        return cls(target, stat_fn, acf_method=acf_method, lags=lags, **configuration_opts)
