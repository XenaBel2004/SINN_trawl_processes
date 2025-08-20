from __future__ import annotations

import torch
from torch import Tensor

from ...utils import BatchedTensorFn
from .normalization import _normalize_data


class CharFunc:
    """Empirical characteristic function estimator (optionally weighted).

    Parameters
    ----------
    data_batched_first
        Shape ``(B, D)`` where ``B`` is the batch size and ``D`` the number of
        selected observation times.
    kernel
        Importance‑sampling kernel ``k(θ)``.  By default it returns a vector of
        ones (i.e. no weighting). θ must be tensor of shape (M, D) where M is
        batch dimension.

    """

    def __init__(
        self,
        data: Tensor,
        *,
        data_batch_first: bool = False,
        kernel: BatchedTensorFn = lambda theta: torch.ones(theta.shape[1], device=theta.device),  # type: ignore
    ) -> None:
        self.data = _normalize_data(data, data_batch_first)  # (B, D, 1)
        self.kernel = kernel

    def __call__(self, theta: Tensor, theta_batch_first: bool = True) -> Tensor:
        """Return the (kernel‑weighted) empirical characteristic function at
        ``θ``.

        Parameters
        ----------
        theta : Tensor
            Shape ``(D, M)`` where ``D`` is the dimensionality of the idx
            and ``M`` the number of Monte‑Carlo points.

        Returns
        -------
        Tensor
            Shape ``(M,)`` – the empirical CF evaluated at each MC point.

        """
        if not theta_batch_first:
            theta = theta.t()
        kern = self.kernel(theta)  # (M,)
        cumulant = 1.0j * (self.data[:, :, 0] @ theta.t())  # (B, M)
        charfunc = torch.mean(torch.exp(cumulant), dim=0)  # (M,)  complex
        return charfunc * kern
