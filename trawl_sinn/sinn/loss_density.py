from __future__ import annotations

from typing import Optional
from .base_loss import BaseStatLoss
from ..processes import StationaryProcessFDD
import numpy as np
import torch
from torch import Tensor

# --------------------------------------------------------------------------- #
#  Utility functions (statistics)
# --------------------------------------------------------------------------- #


def gaussian_kde(
    x: Tensor,
    lower: float,
    upper: float,
    n: int,
    bw: Optional[float] = None,
) -> Tensor:
    """
    One‑dimensional Gaussian kernel density estimator on a regular grid.

    Parameters
    ----------
    x : Tensor
        Arbitrary shape – will be ravelled.
    lower, upper : float
        Grid limits.
    n : int
        Number of grid points.
    bw : float, optional
        Bandwidth. If ``None`` a rule‑of‑thumb ``N^{-1/5}`` is used.

    Returns
    -------
    Tensor
        Shape ``(n,)`` – KDE evaluated on the grid.
    """
    x = x.ravel()
    grid = torch.linspace(lower, upper, steps=n, device=x.device)

    if bw is None:
        bw = float(len(x)) ** (-1 / 5)

    # Kernel matrix: (N, n)
    kernel = torch.exp(-0.5 * ((x[:, None] - grid[None, :]) / bw) ** 2)
    norm = (2 * np.pi) ** 0.5 * len(x) * bw
    density = kernel.sum(dim=0) / norm
    return density


class DensityLoss(BaseStatLoss):
    """Gaussian KDE based density loss."""

    @classmethod
    def analytical(
        cls,
        distr: StationaryProcessFDD,
        lower: float,
        upper: float,
        n: int,
        *,
        bw: Optional[float] = None,
        **configuration_opts,
    ) -> "DensityLoss":
        target = distr.process.pdf(torch.linspace(lower, upper, steps=n))
        def stat_fn(x):
            return gaussian_kde(x, lower=lower, upper=upper, n=n, bw=bw)
        return cls(
            target, stat_fn, lower=lower, upper=upper, n=n, bw=bw, **configuration_opts
        )

    @classmethod
    def empirical(
        cls,
        data: Tensor,
        lower: float,
        upper: float,
        n: int,
        *,
        bw: Optional[float] = None,
        **configuration_opts,
    ) -> "DensityLoss":
        """
        Build a density loss from observed samples.

        Parameters
        ----------
        data
            Samples drawn from the underlying distribution (any shape).
        lower, upper
            Domain of the KDE.
        n
            Number of grid points.
        bw
            Bandwidth; if ``None`` a simple ``N^{-1/5}`` rule is used.
        loss, reduction, kwargs
            Same as in :class:`BaseStatLoss`.
        """
        target = gaussian_kde(data, lower=lower, upper=upper, n=n, bw=bw)
        def stat_fn(x):
            return gaussian_kde(x, lower=lower, upper=upper, n=n, bw=bw)
        return cls(
            target, stat_fn, lower=lower, upper=upper, n=n, bw=bw, **configuration_opts
        )


__all__ = [
    "DensityLoss",
    "gaussian_kde",
]
