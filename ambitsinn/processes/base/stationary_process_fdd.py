# -*- coding: ascii -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
from torch import Generator, Tensor

if TYPE_CHECKING:  # pragma: no cover
    from .stationary_stochastic_process import StationaryStochasticProcess


class StationaryProcessFDD(ABC):
    """Abstract helper that knows the observation grid and the parent process.

    All computationally heavy operations - cumulant/characteristic-function
    evaluation and sampling - are implemented by concrete subclasses.
    """

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------
    def __init__(
        self,
        times: Tensor,
        process: "StationaryStochasticProcess",
        rng: Optional[Generator] = None,
    ):
        self.times: Tensor = times
        self.process: "StationaryStochasticProcess" = process
        self.rng: Generator | None = rng

        if self.process.arg_check:
            self.process._validate_times(times)

    @abstractmethod
    def cumulant(
        self,
        theta: Tensor,
        theta_batch_first: bool = True,
    ) -> Tensor:
        r"""Joint cumulant :math: `\\kappa(\\theta)` for the vector :math:``(X_{t_1},
        ..., X_{t_D})``.

        The implementation must accept ``theta`` either in the internal
        ``(B, D)`` layout (``theta_batch_first=True``) or in the transposed
        layout ``(D, B)`` (``theta_batch_first=False``).  The method should
        normalise the tensor to ``(B, D)`` internally before computation.

        The return tensor must have shape ``(B,)``, where ``B`` is the batch
        dimension supplied with ``theta`` (or implicitly ``1`` if no batch was
        provided).

        Parameters
        ----------
        theta : Tensor
            Fourier-argument tensor; accepted layout is ``(B, D)`` or
            ``(D, B)`` depending on ``theta_batch_first``.
        theta_batch_first : bool, optional
            Specifies the orientation of ``theta``.  ``True`` (default) indicates
            a ``(B, D)`` layout.

        Returns
        -------
        Tensor
            Cumulant values, one per batch.

        """
        raise NotImplementedError

    def charfunc(
        self,
        theta: Tensor,
        theta_batch_first: bool = True,
    ) -> Tensor:
        r"""Characteristic function :math: `\\varphi(\\theta) = exp(\\kappa(\\theta))`.

        This convenience wrapper simply exponentiates the result of
        :meth:`cumulant`.  The same orientation rules for ``theta`` apply.

        Parameters
        ----------
        theta : Tensor
            Fourier-argument tensor (same layout as :meth:`cumulant`).
        theta_batch_first : bool, optional
            Orientation flag for ``theta`` (default: ``True``).

        Returns
        -------
        Tensor
            Characteristic-function values, one per batch.

        """
        return torch.exp(self.cumulant(theta, theta_batch_first))

    @abstractmethod
    def sample(
        self,
        batch_size: int = 1,
        batch_first: Optional[bool] = None,
        unsqueeze_last: Optional[bool] = None,
    ) -> Tensor:
        """Draw trajectories for the current grid.

        Parameters
        ----------
        batch_size : int, default=1
            Number of independent copies to generate.
        batch_first : bool, default=True
            If ``True`` (default) the batch dimension appears first in the
            returned tensor.  If ``False`` the batch dimension is moved to the
            last axis.
        unsqueeze_last : bool, default=True
            If ``True`` (default) a singleton dimension is appended at the end,
            matching the original library convention.
        **kwargs
            Implementation-specific keyword arguments (e.g. a random seed,
            approximation tolerances, etc.).

        Returns
        -------
        Tensor
            Sampled trajectories, of shape ``(batch_size, D, 1)`` (or
            ``(D, batch_size, 1)`` when ``batch_first=False``) unless
            ``unsqueeze_last=False``.

        """
        raise NotImplementedError
