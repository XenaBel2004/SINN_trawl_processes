# -*- coding: ascii -*-
from __future__ import annotations

from abc import abstractmethod
from typing import Optional, cast

import torch
from torch import Generator, Tensor

from ..base import StationaryProcessFDD
from .trawl_process import TrawlProcess


class TrawlProcessFDD(StationaryProcessFDD):
    """Finite-dimensional distribution (FDD) for a generic trawl process.

    It stores the observation times, a reference to the parent process and the *slice-
    measure* matrix needed for cumulant evaluation.
    """

    def __init__(
        self,
        times: Tensor,
        process: TrawlProcess,
        rng: Optional[Generator] = None,
    ) -> None:
        """Parameters
        ----------
        times
            Observation grid (must be strictly increasing).
        process
            Parent :class:`TrawlProcess` instance.
        rng
            Optional generator for deterministic sampling.

        """
        super().__init__(times, process, rng=rng)
        self.slices: Tensor = process._compute_slice_partition(self.times)

        # TODO: implenet proper MyPy fix
        # The following is just hints mypy that super().__init__
        # does not change type of process
        self.process: TrawlProcess = cast(TrawlProcess, self.process)

    # -----------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------
    @abstractmethod
    def _sample_slices(self, batch_size: int = 1) -> Tensor:
        """Sample the random *slice* variables that drive the trawl process.

        Concrete subclasses implement the sampling strategy appropriate for the
        underlying Levy seed (Gaussian, compound Poisson, etc.).

        Parameters
        ----------
        batch_size
            Number of independent copies to generate.

        Returns
        -------
        Tensor
            Tensor of shape ``(B, D, D)`` where the entry
            ``[b, i, j]`` corresponds to the random increment associated
            with the intersection of the trawl sets :math: `A_{t_i}` and :math: `A_{t_j}`.

        """
        raise NotImplementedError

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------
    def cumulant(
        self,
        theta: Tensor,
        theta_batch_first: Optional[bool] = None,
    ) -> Tensor:
        r"""Compute the joint cumulant :math: `\kappa(\theta)` for the vector :math:
        `(X_{t_1}, ..., X_{t_D})` associated with this FDD.

        The implementation follows the textbook formula

        .. math::
            \kappa(\theta) = \sum_{i,j} \ell\bigl(\theta_{i,j}^{+}\bigr) \cdot S_{i,j}

        where :math: `\ell` is the Levy-seed cumulant, :math: `\theta_{i,j}^{+}`
        the cumulative theta matrix (computed by :meth:`TrawlProcess._compute_cumulative_theta`) and
        ``S`` the slice-measure matrix.

        Parameters
        ----------
        theta
            Fourier arguments (shape ``(B, D)`` or ``(D, B)``).
        theta_batch_first
            Overrides the default orientation stored on the parent process.

        Returns
        -------
        Tensor
            Cumulant values of shape ``(B,)``.

        """
        # Normalise theta to ``(B, D)`` layout.
        if self.process.arg_check:
            self.process._validate_cumulant_args(theta, self.times)

        theta_ = self.process._normalize_theta(theta, theta_batch_first)

        # Compute the lower-triangular cumulative theta matrix, evaluate the
        # Levy-seed cumulant on it and weight by the slice-measure matrix.
        cumulants = self.process.seed_cumulant(self.process._compute_cumulative_theta(theta_)) * self.slices
        # Sum over the two slice dimensions, leaving only the batch axis.
        return torch.sum(cumulants, dim=tuple(range(1, cumulants.ndim)))

    def charfunc(
        self,
        theta: Tensor,
        theta_batch_first: Optional[bool] = None,
    ) -> Tensor:
        r"""Characteristic function :math: `\varphi(\theta) =
        exp(\kappa(\theta))`.
        """
        return torch.exp(self.cumulant(theta, theta_batch_first))

    def sample(
        self,
        batch_size: int = 1,
        batch_first: Optional[bool] = None,
        unsqueeze_last: Optional[bool] = None,
    ) -> Tensor:
        """Draw trajectories for the current observation grid.

        The construction proceeds by first sampling the *slice* variables
        (via :meth:`_sample_slices`) and then aggregating them in a cumulative
        fashion to obtain the process values at each observation time.

        Parameters
        ----------
        batch_size
            Number of independent copies to generate.
        batch_first
            If ``True`` (default) the output tensor has shape ``(B, D, 1)``;
            otherwise the shape is ``(D, B, 1)``.
        unsqueeze_last
            If ``True`` (default) a trailing singleton dimension is kept,
            matching the original library's output format.

        Returns
        -------
        Tensor
            Sampled trajectories of shape ``(B, D, 1)`` (or transposed
            when ``batch_first=False``).

        """
        sec_length = self.times.shape[0]

        # Sample the ``(batch_size, D, D)`` slice matrix.
        slices_vals = self._sample_slices(batch_size=batch_size)

        # Initialise an empty tensor for the trajectories.
        traj = torch.zeros((batch_size, sec_length), dtype=slices_vals.dtype, device=slices_vals.device)

        # For each observation time we accumulate the contributions of all slices
        # that intersect the corresponding trawl set.
        for i in range(sec_length):
            # ``slices_vals[:, i:, 0 : i + 1]`` extracts the appropriate lower-triangular
            # block for time ``i``; summing twice yields the sum over both dimensions.
            traj[:, i] = (slices_vals[:, i:, 0 : i + 1]).sum(dim=1).sum(dim=1)

        return self.process._normalize_sample(traj, sample_batch_first=batch_first, sample_unsqueeze=unsqueeze_last)
