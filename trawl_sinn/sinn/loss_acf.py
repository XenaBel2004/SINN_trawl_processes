from __future__ import annotations

from typing import (
    Iterable,
    Optional,
    Union,
    Literal,
)

import torch
from torch import Tensor, IntTensor

from ..processes import StationaryProcessFDD
from .base_loss import BaseStatLoss


# --------------------------------------------------------------------------- #
#  Utility functions (statistics)
# --------------------------------------------------------------------------- #


def acf_fft(x: Tensor, lags: IntTensor) -> Tensor:
    """
    Autocorrelation function using the FFT trick.

    Parameters
    ----------
    x : Tensor
        Shape ``(D, B, ...)``.
    lags : int
        Number of lags to return (``0 … lags-1``).

    Returns
    -------
    Tensor
        Shape ``(lags, ...)`` – normalised autocorrelation for each batch /
        variable pair.
    """

    # Demean the time series
    x_centered = x - x.mean()

    # Zero‑padding to avoid circular convolution artefacts
    n = x_centered.shape[0]
    n_fft = n * 2 - 1

    # FFT, multiply by complex conjugate, inverse FFT → autocovariance
    f = torch.fft.fft(x_centered, n=n_fft, dim=0)
    acov = torch.fft.ifft(f * f.conj(), dim=0).real[:n].mean(axis=1)

    # Normalise – the zero‑lag term is the variance
    acf = acov[lags, ...] / acov[0]
    return acf.squeeze()


def acf_bruteforce(x: Tensor, lags: IntTensor) -> Tensor:
    """
    Direct (O(N·L)) autocorrelation for an arbitrary list of lags.

    Parameters
    ----------
    x : Tensor
        Shape ``(D, B, ...)``.
    lags : int or iterable of int
        If an int, compute ``0 … lags-1``; otherwise a custom list.

    Returns
    -------
    Tensor
        Shape ``(len(lags))``.
    """
    # Demean the time series
    x_centered = x - x.mean()

    out = torch.empty((lags.shape[0], *x.shape[2:]), dtype=x.dtype, device=x.device)

    for i, lag in enumerate(lags):
        if lag == 0:
            out[i] = 1.0
        else:
            # Overlap the two shifted series
            u = x_centered[:-lag, ...]
            v = x_centered[lag:, ...]
            out[i] = torch.sum(u * v, dim=[0, 1]) / torch.sqrt(
                torch.sum(torch.square(u), dim=[0, 1])
                * torch.sum(torch.square(v), dim=[0, 1])
            )
    return out


def _lags_to_idx_tensor(
    lags: Union[int, Iterable[int]], device: Optional[torch.Device] = None
) -> IntTensor:
    if isinstance(lags, int):
        return torch.arange(lags, device=device, dtype=torch.int32)  # type: ignore
    else:
        return torch.tensor(list(lags), device=device, dtype=torch.int32)  # type: ignore


def acf(
    x: Tensor,
    lags: Optional[Union[int, Iterable[int], IntTensor]] = None,
    method: Literal["fft", "brute"] = "fft",
) -> Tensor:
    if lags is None:
        lags = x.shape[0]
    if not isinstance(lags, IntTensor):
        lags = _lags_to_idx_tensor(lags, device=x.device)

    if method == "fft":
        return acf_fft(x, lags)
    if method == "brute":
        return acf_bruteforce(x, lags)


class ACFLoss(BaseStatLoss):
    """Autocorrelation loss"""

    @classmethod
    def analytical(
        cls,
        distr: StationaryProcessFDD,
        lags: Optional[Union[int, Iterable[int], IntTensor]] = None,
        *,
        acf_method: Literal["fft", "brute"] = "fft",
        **configuration_opts,
    ) -> "ACFLoss":
        if lags is None:
            lags = distr.times.shape[0]
        if not isinstance(lags, IntTensor):
            lags = _lags_to_idx_tensor(lags, device=distr.process.device)
        target = distr.process.acf(distr.times[lags])
        def stat_fn(x: Tensor) -> Tensor:
            return acf(x, lags, method=acf_method)
        return cls(
            target, stat_fn, acf_method=acf_method, lags=lags, **configuration_opts
        )

    @classmethod
    def empirical(
        cls,
        data: Tensor,
        lags: Optional[Union[int, Iterable[int], IntTensor]] = None,
        *,
        acf_method: Literal["fft", "brute"] = "fft",
        **configuration_opts,
    ) -> "ACFLoss":
        """
        Build an :class:`ACFLoss` from a set of observed trajectories.

        Parameters
        ----------
        data : Tensor
            Empirical trajectories ``(D, B, T)``.
        lags : int
            Number of lags to include in the target.
        kwargs
            Passed to the underlying :class:`BaseStatLoss`.
        """
        if lags is None:
            lags = data.shape[0]
        if not isinstance(lags, IntTensor):
            lags = _lags_to_idx_tensor(lags, device=data.device)
        target: Tensor = acf(data, lags, method=acf_method)
        def stat_fn(x: Tensor) -> Tensor:
            return acf(x, lags, method=acf_method)
        return cls(
            target, stat_fn, acf_method=acf_method, lags=lags, **configuration_opts
        )


__all__ = [
    "ACFLoss",
    "acf",
]
