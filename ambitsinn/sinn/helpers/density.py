from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

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
        x = x.ravel()

        # choose bandwidth
        bw = self.bw
        if bw is None:
            bw = float(x.numel()) ** (-1 / 5)

        grid = torch.linspace(self.lower, self.upper, steps=self.n, device=x.device)
        kernel = torch.exp(-0.5 * ((x[:, None] - grid[None, :]) / bw) ** 2)
        norm = (2 * math.pi) ** 0.5 * x.numel() * bw
        density = kernel.sum(dim=0) / norm
        return density
