from __future__ import annotations

from typing import Iterable, List, Literal, Optional, Union

from torch import IntTensor, LongTensor, Tensor

from ...processes import StationaryProcessFDD
from ..helpers import ACF, _lags_to_idx_tensor
from .base_loss import BaseStatLoss


class ACFLoss(BaseStatLoss):
    """Autocorrelation loss."""

    @classmethod
    def analytical(
        cls,
        distr: StationaryProcessFDD,
        lags: Optional[Union[int, Iterable[int], IntTensor, LongTensor]] = None,
        *,
        acf_method: Literal["fft", "brute"] = "fft",
        **configuration_opts,
    ) -> "ACFLoss":
        if lags is None:
            lags = distr.times.numel()

        lags_idx: List[int] = _lags_to_idx_tensor(lags) if not isinstance(lags, List) else lags

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
        lags: Optional[Union[int, Iterable[int], IntTensor, LongTensor]] = None,
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

        stat_fn = ACF(
            lags,
            method=acf_method,
            data_batch_first=True,
        )
        target = stat_fn(data)
        return cls(target, stat_fn, acf_method=acf_method, lags=lags, **configuration_opts)
