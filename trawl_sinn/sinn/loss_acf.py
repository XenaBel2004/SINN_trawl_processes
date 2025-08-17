from __future__ import annotations

from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

from ..processes import StationaryStochasticProcess
from .base_loss import *
# --------------------------------------------------------------------------- #
#  Utility functions (statistics)
# --------------------------------------------------------------------------- #


def acf_fft(x: Tensor, lags: int) -> Tensor:
    """
    Autocorrelation function using the FFT trick.

    Parameters
    ----------
    x : Tensor
        Shape ``(T, B, D)``.
    lags : int
        Number of lags to return (``0 … lags-1``).

    Returns
    -------
    Tensor
        Shape ``(lags, B, D)`` – normalised autocorrelation for each batch /
        variable pair.
    """
    # Demean the time series
    x_centered = x - x.mean(dim=0, keepdim=True)

    # Zero‑padding to avoid circular convolution artefacts
    n = x_centered.shape[0]
    n_fft = n * 2 - 1

    # FFT, multiply by complex conjugate, inverse FFT → autocovariance
    f = torch.fft.fft(x_centered, n=n_fft, dim=0)
    acov = torch.fft.ifft(f * f.conj(), dim=0).real[:n].mean(dim = 1)

    # Normalise – the zero‑lag term is the variance
    acf = acov[:lags,...] / acov[0]
    return acf


def acf_bruteforce(x: Tensor, lags: Union[int, Iterable[int]]) -> Tensor:
    """
    Direct (O(N·L)) autocorrelation for an arbitrary list of lags.

    Parameters
    ----------
    x : Tensor
        Shape ``(T, B, D)``.
    lags : int or iterable of int
        If an int, compute ``0 … lags-1``; otherwise a custom list.

    Returns
    -------
    Tensor
        Shape ``(len(lags), B, D)``.
    """
    if isinstance(lags, int):
        lag_idx = torch.arange(lags, device=x.device, dtype=torch.long)
    else:
        lag_idx = torch.tensor(list(lags), device=x.device, dtype=torch.long)

    T, B, D = x.shape
    out = torch.empty((len(lag_idx), B, D), dtype=x.dtype, device=x.device)

    # Demean the time series
    x_centered = x - x.mean(dim=0, keepdim=True)

    # Normalisation term – variance for each (B, D) pair
    var = torch.mean(x_centered**2, dim=0, keepdim=True)  # shape (1, B, D)

    for i, lag in enumerate(lag_idx):
        if lag == 0:
            corr = torch.mean(
                x_centered * x_centered, dim=0
            )  # variance again, same as var.squeeze()
        else:
            # Overlap the two shifted series
            u = x_centered[:-lag]
            v = x_centered[lag:]
            corr = torch.mean(u * v, dim=0)
        out[i] = corr / var.squeeze(0)
    return out


class ACFLoss(BaseStatLoss):
    """FFT‑based autocorrelation loss (deterministic, all lags)."""

    @staticmethod
    def _make_loss_fn(method: str, lags: int):
        if method == "fft":
            stat_fn = lambda x: acf_fft(x, lags)
        elif method == "brute":
            stat_fn = lambda x: acf_bruteforce(x, lags)
        return stat_fn

    @classmethod
    def from_fdd(
        cls,
        process: StationaryStochasticProcess,
        times: torch.Tensor,
        *,
        loss: Union[LossFn, str] = "mse_loss",
        reduction: str = "mean",
        acf_method: str = "fft",
    ) -> "ACFLoss":
        target = process.acf(times)
        stat_fn = cls._make_loss_fn(acf_method, times.shape[0])
        return cls(target, stat_fn, pointwise_loss=loss, reduction=reduction)

    @classmethod
    def from_empirical(
        cls,
        data: Tensor,
        lags: int,
        *,
        loss: Union[LossFn, str] = "mse_loss",
        reduction: str = "mean",
        acf_method: str = "fft",
    ) -> "ACFLoss":
        """
        Build an :class:`ACFLoss` from a set of observed trajectories.

        Parameters
        ----------
        data : Tensor
            Empirical trajectories ``(T, B, D)``.
        lags : int
            Number of lags to include in the target.
        loss : loss function or list thereof (default ``"mse_loss"``)
        reduction : {"mean","sum","none"} (default ``"mean"``)
        kwargs
            Passed to the underlying :class:`BaseStatLoss`.
        """
        target = acf_fft(data, lags)
        stat_fn = cls._make_loss_fn(acf_method, lags)
        return cls(target, stat_fn, pointwise_loss=loss, reduction=reduction)


__all__ = [
    "ACFLoss",
    "acf_bruteforce",
    "acf_fft",
]
