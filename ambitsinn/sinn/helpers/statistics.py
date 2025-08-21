import math
from typing import List, Optional

import torch
from torch import Tensor


def acf_fft(data: Tensor, lags: List[int]) -> Tensor:
    """
    Parameters:
    -----------
    data
        Tensor of shape (B,D,1)
    lags
        List of postive lags, maximum of which does not exceed D

    Returns
    -------
    acf
        1-D tensor of shape (len(lags))
    """

    # Zero‑padding to avoid circular convolution artefacts
    n = data.shape[1]
    n_fft = n * 2 - 1

    # FFT, multiply by complex conjugate, inverse FFT -> autocovariance
    f = torch.fft.fft(data[:, :, 0] - data.mean(), n=n_fft, dim=1)
    acov = torch.fft.ifft(f * f.conj(), dim=1).real[:, :n].mean(dim=0)

    # Normalise
    acf = acov[lags] / acov[0]

    return acf


def acf_bruteforce(data: Tensor, lags: List[int]) -> Tensor:
    """Direct (O(N·L)) autocorrelation for an arbitrary list of lags.

    Parameters
    ----------
    data
        Shape ``(D, B, 1)``.

    lags
        List of postive lags, maximum of which does not exceed D

    Returns
    -------
    Tensor
        Shape ``(len(lags))``.

    """
    # Demean the time series
    data = data - data.mean()

    # Allocate memory
    acf = torch.empty(len(lags), dtype=data.dtype, device=data.device)

    for i, lag in enumerate(lags):
        if lag == 0:
            acf[i] = 1.0
        else:
            # Overlap the two shifted series
            u = data[:, :-lag, :]
            v = data[:, lag:, :]
            # Classical ACF estimator
            acf[i] = torch.sum(u * v) / torch.sqrt(torch.sum(torch.square(u)) * torch.sum(torch.square(v)))

    return acf


def gaussian_kde(data: Tensor, lower: float, upper: float, n: int, bw: Optional[float] = None) -> Tensor:
    # Normalize input data and create grid
    data = data.ravel()
    bw = bw or float(data.numel()) ** (-1 / 5)
    grid = torch.linspace(lower, upper, n, device=data.device)

    # Compute KDE
    kern = torch.exp(-0.5 * ((data[:, None] - grid[None, :]) / bw) ** 2)
    norm = (2 * math.pi) ** 0.5 * data.numel() * bw
    density = kern.sum(dim=0) / norm

    return density


def charfunc(data: Tensor, theta: Tensor) -> Tensor:
    """
    Parameters:
    -----------
    data
        Tensor of shape (B,D,1)
    theta
        Tensor of shape (M, D)

    Returns
    -------

    charfunc
        Tensor of shape (M)
    """
    return torch.mean(
        torch.exp(
            1.0j * (data * theta.t())  # (B, M)
        ),
        dim=0,
    )
