from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ...processes import StationaryProcessFDD
from ..helpers import GaussianKDE
from .base_loss import BaseStatLoss

# --------------------------------------------------------------------------- #
#  Utility functions (statistics)
# --------------------------------------------------------------------------- #


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
    ) -> DensityLoss:
        stat_fn = GaussianKDE(lower=lower, upper=upper, n=n, bw=bw)
        target = distr.process.pdf(torch.linspace(lower, upper, steps=n, device=distr.process.device))
        return cls(target, stat_fn, lower=lower, upper=upper, n=n, bw=bw, **configuration_opts)

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
        """Build a density loss from observed samples.

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
        stat_fn = GaussianKDE(lower=lower, upper=upper, n=n, bw=bw)
        target = stat_fn(data)
        return cls(target, stat_fn, lower=lower, upper=upper, n=n, bw=bw, **configuration_opts)
