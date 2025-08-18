# -*- coding: ascii -*-
"""
Concrete implementation of a *Gaussian* trawl process.

The Levy seed is a standard Gaussian random measure, so the cumulant of the
seed is
.. math::
    \\ell(\\theta) = i \\mu \\theta - \\frac12 \\sigma^2 \\theta^2

with :math: `\\mu` and :math: `\\sigma^2` derived from the marginal mean/variance of the
process and the area of the trawl set.

The class inherits all heavy-lifting from :class:`TrawlProcess` and
provides a closed-form marginal density function.
"""

import torch
from torch import Tensor
from .trawl import TrawlProcess, TrawlProcessFDD
from typing import Callable, Optional, cast


class GaussianTrawlProcess(TrawlProcess):
    """
    Gaussian trawl process ``X_t = L(A_t)`` where the Levy seed ``L'`` is
    Gaussian.  The marginal distribution of ``X_t`` is therefore also Gaussian
    with user-specified mean and variance.

    Parameters
    ----------
    integrated_trawl_function
        Callable implementing the integrated trawl function :math:`G`.
    mean
        Desired marginal mean of ``X_t``.
    var
        Desired marginal variance of ``X_t``.
    theta_batch_first, arg_check
        Forwarded to :class:`TrawlProcess`.
    """

    def __init__(
        self,
        integrated_trawl_function: Callable[[Tensor], Tensor],
        mean: float = 0.0,
        var: float = 1.0,
        **default_opts,
    ) -> None:
        # Store scalar parameters as zero-dimensional tensors (no gradient)
        self.mean: Tensor = torch.tensor(mean, device=default_opts.get("device"))
        self.var: Tensor = torch.tensor(var, device=default_opts.get("device"))
        self.std: Tensor = torch.sqrt(self.var)

        # Area of the trawl set at lag zero (used to scale seed moments)
        self.trawl_area: Tensor = integrated_trawl_function(torch.tensor(0.0))

        # Parameters of the Levy **seed** (i.e. of ``L'``)
        # mu_seed = mean / |A_0|
        self.seed_mean: Tensor = self.mean / self.trawl_area
        # var_seed = var / |A_0|
        self.seed_var: Tensor = self.var / self.trawl_area
        self.seed_std: Tensor = torch.sqrt(self.seed_var)

        # -----------------------------------------------------------------
        # Levy-seed cumulant for a *Gaussian* seed.
        # -----------------------------------------------------------------
        def seed_cumulant(theta: Tensor) -> Tensor:
            """
            Log-characteristic function (cumulant) of the Gaussian Levy seed.

            Parameters
            ----------
            theta
                Tensor of Fourier arguments (real-valued).

            Returns
            -------
            Tensor
                Complex-valued cumulant :math: `i \\cdot \\mu \\cdot \\theta + \\frac12 \\cdot \\sigma^2 \\cdot \\theta^2`.
            """
            # Ensure the computation is performed in a complex dtype.
            theta_c = theta.to(
                dtype=torch.complex64
                if theta.dtype in (torch.float32, torch.complex64)
                else torch.complex128
            )
            return 1.0j * self.seed_mean * theta_c - 0.5 * (self.seed_std**2) * (
                theta_c**2
            )

        super().__init__(seed_cumulant, integrated_trawl_function, **default_opts)

    def pdf(self, x: Tensor) -> Tensor:
        """
        Marginal Gaussian density of the process.

        Parameters
        ----------
        x
            Points at which to evaluate the pdf.

        Returns
        -------
        Tensor
            Density values (same shape as ``x``).
        """
        return torch.distributions.normal.Normal(self.mean, self.std).log_prob(x).exp()

    def at_times(
        self, times: Tensor, rng: Optional[torch.Generator] = None
    ) -> "GaussianTrawlProcessFDD":
        """
        Build the finite-dimensional distribution object for a given grid.

        Parameters
        ----------
        times
            Observation times (strictly increasing 1-D tensor).
        rng
            Optional random-number generator for reproducible sampling.

        Returns
        -------
        GaussianTrawlProcessFDD
            The concrete FDD bound to ``self`` and ``times``.
        """
        return GaussianTrawlProcessFDD(times, self, rng)


class GaussianTrawlProcessFDD(TrawlProcessFDD):
    """
    Finite-dimensional distribution for :class:`GaussianTrawlProcess`.

    It implements the Gaussian-seed sampling strategy required by the base
    class.
    """

    def __init__(
        self,
        times: Tensor,
        process: GaussianTrawlProcess,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        """
        Parameters
        ----------
        times
            Observation grid.
        process
            Parent :class:`GaussianTrawlProcess` instance.
        rng
            Optional generator for deterministic sampling.
        """
        super().__init__(times, process, rng=rng)
        self.process: GaussianTrawlProcess = cast(
            GaussianTrawlProcess, self.process
        )  # MyPy fix

    def _sample_slices(self, batch_size: int = 1) -> Tensor:
        """
        Sample the Gaussian slice variables.

        For a Gaussian Levy seed the slice values are independent Gaussian
        random variables with means and variances proportional to the slice
        measures.

        Parameters
        ----------
        batch_size
            Number of independent trajectories to sample.

        Returns
        -------
        Tensor
            Tensor of shape ``(batch_size, D, D)`` containing the sampled
            slice increments.
        """
        # ``self.slices`` has shape (D, D).  Broadcast to the batch dimension.
        slices_ = self.slices.broadcast_to(
            batch_size, self.slices.shape[0], self.slices.shape[0]
        )
        # Allocate an array of zeros with the same dtype/device as ``slices_``.
        result = torch.zeros_like(slices_)

        # Only the lower-triangular entries are non-zero (by construction);
        # sample those entries from the appropriate Gaussian distribution.
        mask = slices_ != 0
        result[mask] = torch.normal(
            mean=slices_[mask] * self.process.seed_mean,
            std=torch.sqrt(slices_[mask] * self.process.seed_var),
            generator=self.rng,
        )
        return result
