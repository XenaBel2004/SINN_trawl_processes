# gaussian_trawl.py
"""
Concrete implementation of a *Gaussian* trawl process.

The Lévy seed is a standard Gaussian random measure, so the cumulant of the
seed is

    ℓ(u) = i μ u - ½ σ² u²

with ``μ`` and ``σ²`` derived from the marginal mean/variance of the
process and the area of the trawl set.

The class inherits all heavy‑lifting from :class:`TrawlProcess` and
provides a closed‑form marginal density function.
"""

import torch
from torch import Tensor
from .trawl import TrawlProcess, TrawlProcessFDD
from typing import Callable, Optional


class GaussianTrawlProcess(TrawlProcess):
    """
    Gaussian trawl process ``X_t = L(A_t)`` where the Lévy seed ``L'`` is
    Gaussian.  The marginal distribution of ``X_t`` is therefore also Gaussian
    with user‑specified mean and variance.

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
        theta_batch_first: bool = True,
        arg_check: bool = True,
    ) -> None:
        # Store scalar parameters as zero‑dimensional tensors (no gradient)
        self.mean: Tensor = torch.tensor(mean, requires_grad=False)
        self.var: Tensor = torch.tensor(var, requires_grad=False)
        self.std: Tensor = torch.sqrt(self.var)

        # Area of the trawl set at lag zero (used to scale seed moments)
        self.trawl_area: Tensor = integrated_trawl_function(torch.tensor(0.0))

        # Parameters of the Lévy **seed** (i.e. of ``L'``)
        # The relationship is:   mean_X = μ_seed * |A_0|   →   μ_seed = mean / |A_0|
        self.seed_mean: Tensor = self.mean / self.trawl_area
        #   var_X = σ_seed² * |A_0|   →   σ_seed² = var / |A_0|
        self.seed_var: Tensor = self.var / self.trawl_area
        self.seed_std: Tensor = torch.sqrt(self.seed_var)

        # -----------------------------------------------------------------
        # Lévy‑seed cumulant for a *Gaussian* seed.
        # -----------------------------------------------------------------
        #   ℓ(u) = i μ u - ½ σ² u².
        #   The callable must return a **complex** tensor.
        # -----------------------------------------------------------------
        def seed_cumulant(u: Tensor) -> Tensor:
            """
            Log‑characteristic function (cumulant) of the Gaussian Lévy seed.

            Parameters
            ----------
            u
                Tensor of Fourier arguments (real‑valued).

            Returns
            -------
            Tensor
                Complex‑valued cumulant ``i·μ·u - ½·σ²·u²``.
            """
            # Ensure the computation is performed in a complex dtype.
            u_c = u.to(
                dtype=torch.complex64
                if u.dtype in (torch.float32, torch.complex64)
                else torch.complex128
            )
            return 1.0j * self.seed_mean * u_c - 0.5 * (self.seed_std**2) * (u_c**2)

        super().__init__(
            seed_cumulant,
            integrated_trawl_function,
            arg_check=arg_check,
            theta_batch_first=theta_batch_first,
        )

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
        Build the finite‑dimensional distribution object for a given grid.

        Parameters
        ----------
        times
            Observation times (strictly increasing 1‑D tensor).
        rng
            Optional random‑number generator for reproducible sampling.

        Returns
        -------
        GaussianTrawlProcessFDD
            The concrete FDD bound to ``self`` and ``times``.
        """
        return GaussianTrawlProcessFDD(times, self, rng)


class GaussianTrawlProcessFDD(TrawlProcessFDD):
    """
    Finite‑dimensional distribution for :class:`GaussianTrawlProcess`.

    It implements the Gaussian‑seed sampling strategy required by the base
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
        self.times: Tensor = times
        self.process: GaussianTrawlProcess = process
        self.slices: Tensor = self.process._compute_slice_partition(self.times)
        self.rng: Optional[torch.Generator] = rng

    def _sample_slices(self, batch_size: int = 1) -> Tensor:
        """
        Sample the Gaussian slice variables.

        For a Gaussian Lévy seed the slice values are independent Gaussian
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

        # Only the lower‑triangular entries are non‑zero (by construction);
        # sample those entries from the appropriate Gaussian distribution.
        mask = slices_ != 0
        result[mask] = torch.normal(
            mean=slices_[mask] * self.process.seed_mean,
            std=torch.sqrt(slices_[mask] * self.process.seed_var),
            generator=self.rng,
        )
        return result
