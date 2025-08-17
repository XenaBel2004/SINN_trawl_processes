# gamma_trawl.py
"""
Gamma‑trawl process implementation.

In this file the Lévy seed is a **Gamma** random measure with shape
parameter ``α_seed`` and rate ``β_seed``.  The marginal distribution of
``Xₜ`` is again Gamma, with shape ``α`` and rate ``β`` supplied by the user.
The relationship between the marginal and seed parameters is

    α = α_seed · |A₀|,      β = β_seed,

where ``|A₀| = G(0)`` is the Lebesgue measure of a trawl set at lag zero.

The module provides two concrete classes:

* :class:`GammaTrawlProcess` – the stochastic process object.
* :class:`GammaTrawlProcessFDD` – its finite‑dimensional distribution,
  required for cumulants, characteristic functions and sampling.

Both classes inherit the heavy‑lifting logic from :class:`TrawlProcess` and
:class:`TrawlProcessFDD` defined in ``trawl.py``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Callable, Optional

from .trawl import TrawlProcess, TrawlProcessFDD


class GammaTrawlProcess(TrawlProcess):
    """
    Stationary **Gamma trawl** process ``Xₜ = L(Aₜ)``.

    The Lévy seed ``L′`` is a Gamma random measure with *shape* ``α_seed`` and
    *rate* ``β_seed``.  The marginal distribution of ``Xₜ`` is
    ``Gamma(α, β)`` with user‑specified ``shape`` (α) and ``rate`` (β).

    Parameters
    ----------
    integrated_trawl_function :
        Callable implementing the *integrated* trawl function ``G(s) = ∫_s^∞ g(τ) dτ``.
        It must accept a ``torch.Tensor`` of (possibly negative) time differences
        and return a real‑valued tensor of the same shape.
    shape :
        Desired marginal shape parameter ``α`` of the process (default 1.0).
    rate :
        Desired marginal rate parameter ``β`` of the process (default 1.0).
    theta_batch_first :
        Orientation flag passed to the base class (default ``True``).
    arg_check :
        Whether to run run‑time validation of user inputs (default ``True``).

    Notes
    -----
    Let ``A₀`` denote the trawl set at lag zero; its Lebesgue measure is
    ``|A₀| = G(0)``.  The seed parameters are obtained by normalising the
    marginal parameters:

        α_seed = α / |A₀|,      β_seed = β.

    With these definitions the cumulant (log‑characteristic function) of the
    seed is

        ℓ(u) = -α_seed * log(1 - i u / β_seed),

    which is used throughout the cumulant/characteristic‑function machinery.
    """

    def __init__(
        self,
        integrated_trawl_function: Callable[[Tensor], Tensor],
        shape: float = 1.0,
        rate: float = 1.0,
        theta_batch_first: bool = True,
        arg_check: bool = True,
    ) -> None:
        # -----------------------------------------------------------------
        # Store marginal parameters as zero‑dimensional tensors (no gradient).
        # -----------------------------------------------------------------
        self.shape: Tensor = torch.tensor(shape, requires_grad=False)
        self.rate: Tensor = torch.tensor(rate, requires_grad=False)

        # Area of a single trawl set at lag zero (|A₀| = G(0)).
        self.trawl_area: Tensor = integrated_trawl_function(torch.tensor(0.0))

        # Seed parameters derived from the desired marginal law.
        #   α_seed = α / |A₀|   and   β_seed = β
        self.seed_shape: Tensor = self.shape / self.trawl_area
        self.seed_rate: Tensor = self.rate

        # -----------------------------------------------------------------
        # Lévy‑seed cumulant (log‑characteristic function) for a Gamma seed.
        # -----------------------------------------------------------------
        def seed_cumulant(u: Tensor) -> Tensor:
            """
            Log‑characteristic function ℓ(u) of a Gamma Lévy seed.

            The Gamma seed has shape ``α_seed`` and rate ``β_seed``.  Its
            log‑characteristic function is

                ℓ(u) = -α_seed * log(1 - i u / β_seed).

            Parameters
            ----------
            u : Tensor
                Real‑valued Fourier arguments.  The tensor is cast to a
                complex dtype before the computation.

            Returns
            -------
            Tensor
                Complex‑valued cumulant, broadcastable to the shape of ``u``.
            """
            # Cast to a complex dtype that matches the input precision.
            if u.dtype == torch.float32:
                u_c = u.to(torch.complex64)
            elif u.dtype == torch.float64:
                u_c = u.to(torch.complex128)
            else:
                # If ``u`` is already complex we keep it as is.
                u_c = u

            # The expression 1 - i u/β is safe because β > 0 (enforced by the user).
            return -self.seed_shape * torch.log(1 - 1j * u_c / self.seed_rate)

        # Initialise the abstract base class with the seed cumulant and the
        # integrated trawl function.
        super().__init__(
            seed_cumulant,
            integrated_trawl_function,
            arg_check=arg_check,
            theta_batch_first=theta_batch_first,
        )

    # -----------------------------------------------------------------
    # Public API required by :class:`StationaryStochasticProcess`
    # -----------------------------------------------------------------
    def pdf(self, x: Tensor) -> Tensor:
        """
        Marginal probability density of the Gamma trawl process.

        Parameters
        ----------
        x : Tensor
            Points at which to evaluate the density.

        Returns
        -------
        Tensor
            Density values with the same shape as ``x``.
        """
        return torch.distributions.gamma.Gamma(self.shape, self.rate).log_prob(x).exp()

    def at_times(
        self, times: Tensor, rng: Optional[torch.Generator] = None
    ) -> "GammaTrawlProcessFDD":
        """
        Build the finite‑dimensional distribution (FDD) for a specific
        observation grid.

        Parameters
        ----------
        times :
            Strictly increasing 1‑D tensor of observation times.
        rng :
            Optional ``torch.Generator`` for deterministic sampling.

        Returns
        -------
        GammaTrawlProcessFDD
            The concrete FDD bound to this process and the supplied ``times``.
        """
        return GammaTrawlProcessFDD(times, self, rng)


class GammaTrawlProcessFDD(TrawlProcessFDD):
    """
    Finite‑dimensional distribution for :class:`GammaTrawlProcess`.

    This class implements the slice‑sampling routine required by the generic
    ``TrawlProcessFDD.sample`` algorithm.  All other heavy‑lifting methods
    (cumulant, characteristic function) are inherited unchanged from the
    parent class.
    """

    def __init__(
        self,
        times: Tensor,
        process: GammaTrawlProcess,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        """
        Parameters
        ----------
        times :
            Observation grid (strictly increasing 1‑D tensor).
        process :
            Parent :class:`GammaTrawlProcess` instance.
        rng :
            torch.Generator object, left for compability. Currently, reproducuble sampling
            is possible only via ``torch.manual_seed``
        """
        self.times: Tensor = times
        self.process: GammaTrawlProcess = process
        self.slices: Tensor = self.process._compute_slice_partition(self.times)
        self.rng: Optional[torch.Generator] = rng

    # -----------------------------------------------------------------
    # Internal helper – sampling the Gamma slice variables
    # -----------------------------------------------------------------
    def _sample_slices(self, batch_size: int = 1) -> Tensor:
        """
        Sample a ``(batch_size, D, D)`` tensor of slice values.

        For a Gamma Lévy seed the slice variable associated with a
        Lebesgue measure ``Δ`` follows a Gamma distribution with

            shape  = Δ * α_seed
            rate   = β_seed .

        Only the lower‑triangular entries of ``self.slices`` are non‑zero; those
        entries are sampled, while the remaining entries stay at zero.

        Parameters
        ----------
        batch_size : int, default=1
            Number of independent trajectories to generate.

        Returns
        -------
        Tensor
            Sampled slice matrix of shape ``(batch_size, D, D)`` where
            ``D = len(times)``.
        """
        # Broadcast the slice‑measure matrix to the batch dimension.
        slices_ = self.slices.broadcast_to(
            batch_size, self.slices.shape[0], self.slices.shape[0]
        )
        result = torch.zeros_like(slices_)

        # Boolean mask of entries that need to be sampled (non‑zero Lebesgue measure).
        mask = slices_ != 0

        # Shape parameters for the Gamma distribution, one per non‑zero entry.
        shape_params = slices_[mask] * self.process.seed_shape
        rate_param = self.process.seed_rate

        # Sample from the Gamma distribution.  ``generator`` is not supported by
        # torch yet, so one should use ``torch.manual_seed`` instead
        result[mask] = torch.distributions.gamma.Gamma(
            shape_params,
            rate_param,
        ).sample()

        return result
