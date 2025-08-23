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
        lags = lags or distr.times.numel()
        idxs = _lags_to_idx_tensor(lags) if not isinstance(lags, List) else lags

        target = distr.process.acf(distr.times[idxs])
        stat_fn = ACF(
            idxs, method=acf_method, data_batch_first=True
        )  # due to normalization in forward pass of base stat loss

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
        lags = lags or data.numel()
        idxs = _lags_to_idx_tensor(lags) if not isinstance(lags, List) else lags

        stat_fn = ACF(
            idxs, method=acf_method, data_batch_first=True
        )  # due to normalization in forward pass of base stat loss
        target = stat_fn(data)

        return cls(target, stat_fn, acf_method=acf_method, lags=lags, **configuration_opts)
