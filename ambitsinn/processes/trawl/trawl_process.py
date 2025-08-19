# -*- coding: ascii -*-
r"""Implementation of *trawl* processes - a class of stationary stochastic
processes defined via a Levy basis :math: `L(\cdot)`` and a moving trawl set :math: `A_t`.
The key mathematical objects are:

* ``seed_cumulant`` - a callable implementing the *log characteristic function*
  (i.e. the cumulant) of the Levy seed "math: `L' = L([0; 1] \times [0; 1])`.
* ``integrated_trawl_function`` - the antiderivative of the trawl kernel
  ``g`` (denoted :math:`G(s) = \int_{s}^{+\infty}g(t)dt).

The concrete class :class:`GaussianTrawlProcess` (see ``gaussian_trawl.py``) is the
most common example.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
from torch import Generator, Tensor

from ...utils import BatchedTensorFn, ElementwiseFn
from ..base import StationaryStochasticProcess

if TYPE_CHECKING:  # pragma: no cover
    from .trawl_process_fdd import TrawlProcessFDD


class TrawlProcess(StationaryStochasticProcess):
    r"""Abstract base class for *trawl* processes.

    A trawl process is defined as

    .. math::
        X_t = L(A_t) = \int_{A_t} L(\mathrm{d}u),

    where ``L`` is a Levy basis whose (log) characteristic function is
    supplied via ``seed_cumulant`` and ``A_t`` is a *trawl set* of the form

    .. math::
        A_t = \{(x, t) |0 \leq x \leq g(-t)\}

    governed by the *integrated trawl function* ``G`` such that ``G' = g``.

    Sub-classes must implement a concrete probability density function (``pdf``)
    and a method to construct the finite-dimensional distribution
    (:meth:`at_times`).

    Parameters
    ----------
    seed_cumulant
        Callable returning the *cumulant* of the Levy seed, i.e.

        ``seed_cumulant(u) = log E[exp(i * u * L')]``  (complex-valued).

        The function must accept a ``torch.Tensor`` of arbitrary shape and
        return a tensor of identical shape with ``dtype=torch.complex64`` or
        ``torch.complex128``.

    integrated_trawl_function
        Callable implementing the *integrated trawl function* :math:`G(s)`.
        It should accept a ``torch.Tensor`` (possibly negative) and return a
        real-valued ``torch.Tensor`` of the same shape.

    **default_options
        Forwarded to :class:`StationaryStochasticProcess`.

    """

    def __init__(
        self,
        seed_cumulant: BatchedTensorFn,
        integrated_trawl_function: ElementwiseFn,
        **default_options,
    ) -> None:
        super().__init__(**default_options)
        # ell(u) = log E[exp(i * u * L')]
        self.seed_cumulant: BatchedTensorFn = seed_cumulant
        # G(s) = Integral[s; +inf] g(t) dt
        self.it_func: ElementwiseFn = integrated_trawl_function

    # --------------------------------------------------------------------------
    # slice matrix (matrix of ``lebesgue measure of slices`` at fixed timestamps)
    # --------------------------------------------------------------------------
    def _compute_slice_partition(self, times: Tensor) -> Tensor:
        """Compute the *slice-measure* matrix ``S`` for a given observation grid.

        The matrix ``S`` has shape ``(D, D)`` where ``D = len(times)``.
        Entry ``S[i, j]`` (with ``i <= j``) equals the Lebesgue measure of the
        intersection of the trawl sets :math: `A_{t_i}` and :math: `A_{t_j}`.

        The implementation follows the vectorised ``hack``.  It relies on the fact
        that the measure of the intersection can be expressed using the
        integrated trawl function evaluated at the four corner differences of the
        rectangle formed by the two times.

        Parameters
        ----------
        times
            1-D tensor of increasing observation times.

        Returns
        -------
        Tensor
            Lower-triangular slice-measure matrix of shape ``(D, D)``.

        """
        D = times.shape[0]

        # Time differences required for the four corners of each rectangle.
        # The ``view``/``roll`` gymnastics give us a full pairwise matrix.
        t_diff_j_i = times.view(1, -1) - times.view(-1, 1)  # t_j - t_i
        t_diff_j_im1 = times.view(1, -1) - times.roll(1).view(-1, 1)  # t_j - t_{i-1}
        t_diff_jp1_i = times.roll(-1).view(1, -1) - times.view(-1, 1)  # t_{j+1} - t_i
        t_diff_jp1_im1 = times.roll(-1).view(1, -1) - times.roll(1).view(-1, 1)

        # The ``four-corner`` formula (see the paper for a derivation).
        A = (
            self.it_func(t_diff_jp1_im1)
            - self.it_func(t_diff_jp1_i)
            + self.it_func(t_diff_j_i)
            - self.it_func(t_diff_j_im1)
        )
        # Edge corrections for the first row and last column.
        A[:, D - 1] = self.it_func(t_diff_j_i[:, D - 1]) - self.it_func(t_diff_j_im1[:, D - 1])
        A[0, :] = self.it_func(t_diff_j_i[0, :]) - self.it_func(t_diff_jp1_i[0, :])
        A[0, D - 1] = self.it_func(times[D - 1] - times[0])

        # Transpose and retain the lower-triangular part.
        return torch.tril(A.T)

    def _compute_cumulative_theta(self, theta: Tensor) -> Tensor:
        r"""Compute the *cumulative* version of ``theta`` required for the cumulant
        calculation of a trawl process.

        For a tensor ``theta`` of shape ``(B, D)`` the function returns a
        3-D tensor ``C`` of shape ``(B, D, D)`` where

        .. math::
            C_{b,i,j} = \sum_{k=0}^i \mathbf{1}(k \leq j)\theta_{b,k} .

        In other words, each slice ``C[b]`` contains lower-triangular
        cumulative sums of the corresponding batch ``theta[b]``.

        Parameters
        ----------
        theta
            Tensor of shape ``(B, D)`` (batch-first layout).

        Returns
        -------
        Tensor
            Tensor of shape ``(B, D, D)`` containing the lower-triangular
            cumulative sums.

        """
        B, D = theta.shape
        # unsqueeze: (B, D) -> (B, D, 1);
        # broadcast: (B, D) -> (B, D, D);
        # ``torch.tril(... )`` zeroes the upper-triangular part.
        return torch.tril(theta.unsqueeze(2).broadcast_to(B, D, D)).cumsum(1)

    def acf(self, lags: Tensor) -> Tensor:
        r"""Autocorrelation function for a trawl process.

        The (theoretical) ACF is proportional to the integrated trawl function:

        .. math::
            \rho(t) = \frac{G(t)}{G(0)}.

        Parameters
        ----------
        lags
            Non-negative distance between two time points.

        Returns
        -------
        Tensor
            Autocorrelation values with the same shape as ``lags``.

        """
        return self.it_func(torch.abs(lags)) / self.it_func(torch.tensor(0.0))

    @abstractmethod
    def pdf(self, x: Tensor) -> Tensor:
        """Probability density function - must be implemented by concrete subclasses."""
        raise NotImplementedError

    @abstractmethod
    def at_times(self, times: Tensor, rng: Optional[Generator] = None) -> "TrawlProcessFDD":
        """Create a :class:`TrawlProcessFDD` object for the supplied observation grid.

        Parameters
        ----------
        times
            Tensor of increasing observation times.
        rng
            Optional generator for reproducible sampling.

        Returns
        -------
        TrawlProcessFDD
            Finite-dimensional distribution bound to ``self`` and ``times``.

        """
        raise NotImplementedError
