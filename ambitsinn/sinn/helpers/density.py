from __future__ import annotations

from typing import Optional

from torch import Tensor

from .statistics import gaussian_kde

# --------------------------------------------------------------------------- #
#  Utility functions (statistics)
# --------------------------------------------------------------------------- #


class GaussianKDE:
    def __init__(
        self,
        lower: float,
        upper: float,
        n: int,
        bw: Optional[float] = None,
    ):
        """One‑dimensional Gaussian kernel density estimator on a regular grid.

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
        self.lower = lower
        self.upper = upper
        self.n = n
        self.bw = bw

    def __call__(self, x: Tensor) -> Tensor:
        return gaussian_kde(x, self.lower, self.upper, self.n, self.bw)
