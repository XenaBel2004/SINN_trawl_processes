from __future__ import annotations

from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from .base_loss import LossFn, BaseStatLoss
from ..processes import StationaryStochasticProcess
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

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
    def from_fdd(
        cls,
        process: StationaryStochasticProcess,
        lower: float,
        upper: float,
        n: int,
        *,
        bw: Optional[float] = None,
        loss: Union[LossFn, str] = "mse_loss",
        reduction: str = "mean",
    ) -> "DensityLoss":
        target = process.pdf(torch.linspace(lower, upper, steps=n))
        stat_fn = lambda x: gaussian_kde(x, lower=lower, upper=upper, n=n, bw=bw)
        return cls(
            target,
            stat_fn,
            pointwise_loss=loss,
            reduction=reduction,
            lower=lower,
            upper=upper,
            n=n,
            bw=bw,
        )

    @classmethod
    def from_empirical(
        cls,
        data: Tensor,
        lower: float,
        upper: float,
        n: int,
        *,
        bw: Optional[float] = None,
        loss: Union[LossFn, str] = "mse_loss",
        reduction: str = "mean",
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
        stat_fn = lambda x: gaussian_kde(x, lower=lower, upper=upper, n=n, bw=bw)
        return cls(
            target,
            stat_fn,
            pointwise_loss=loss,
            reduction=reduction,
            lower=lower,
            upper=upper,
            n=n,
            bw=bw,
        )


__all__ = [
    "DensityLoss",
    "gaussian_kde",
]
