# -*- coding: utf-8 -*-
"""
NoiseProcess – a collection of *i.i.d.* random variables.

The process is defined solely by the log‑characteristic function of a single
margin (the *cumulant function*) and, torch Distribution object.  Because the variables are independent, the joint cumulant is simply
the sum of the marginal cumulants and the autocorrelation is 1 at lag 0 and
0 elsewhere.

The implementation conforms to the abstract API defined in ``base.py``:
* ``NoiseProcess`` derives from :class:`StationaryStochasticProcess`.
* ``NoiseFDD`` derives from :class:`StationaryProcessFDD`.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, overload

import torch
from torch import Tensor

# The base abstractions we need to inherit from.
from .base import StationaryStochasticProcess, StationaryProcessFDD


class OUProcess(StationaryStochasticProcess):
    """
    dX_t = = lambda * (mu - X_t) * dt + sigma * dW_t
    https://encyclopediaofmath.org/wiki/Ornstein-Uhlenbeck_process

    X_t is stationary Gaussian process with mean 'mu' and autocovariance
    function
    'autocov(s) = sigma^2 / (2 * lambda) exp(-lambda |s|)'

    parameters:
        lambda_  : Strength of the restoring force
        sigma    : Noise strength
        mu       : Long-term mean
        delta_t  : Discretization step
    """
    def __init__(self,
        mu       : float = 0.0,
        sigma    : float = 1.0,
        lambda_  : float = 1.0,
        *,
        theta_batch_first: bool = True,
        arg_check: bool = True,
    ) -> None:
        super().__init__(arg_check=arg_check, theta_batch_first=theta_batch_first)

        self.lambda_ : Tensor = torch.tensor(lambda_)
        self.sigma  : Tensor = torch.tensor(self.sigma)
        self.mu :Tensor = torch.tensor(mu)

    def at_times(
        self, times: Tensor, rng: Optional[torch.Generator] = None
    ) -> "OUProcessFDD":
        """
        Create a finite‑dimensional distribution (FDD) for the supplied
        observation grid.

        The returned ``NoiseFDD`` knows the grid, the cumulant function,
        and (if supplied) the sampler / PDF.

        Parameters
        ----------
        times :
            Strictly increasing 1‑D tensor of observation times.
        rng :
            Optional ``torch.Generator`` that will be passed to the sampler
            (if it accepts a ``generator`` argument).

        Returns
        -------
        NoiseFDD
            An FDD object that implements :meth:`cumulant`, :meth:`charfunc`,
            and (optionally) :meth:`sample`.
        """
        # Validation of ``times`` lives in the base class – reuse it.
        self._validate_times(times)
        return OUProcessFDD(self, times, rng=rng)

    def pdf(self, x: Tensor) -> Tensor:
        """
        Marginal probability density function.

        Parameters
        ----------
        x :
            Points at which to evaluate the density.

        Returns
        -------
        Tensor
            Density values with the same shape as ``x``.
        """
        return torch.distributions.normal.Normal(
            torch.tensor(self.mu), self.sigma / torch.sqrt(2 * self.lambda_)
        ).log_prob(x).exp()

    def acf(self, lags: Tensor) -> Tensor:
        """
        Autocorrelation function for an i.i.d. process.

        By definition the autocorrelation equals ``1`` at lag ``0`` and ``0``
        for any non‑zero lag.

        Parameters
        ----------
        lags :
            Tensor of non‑negative lags (same shape as the return value).

        Returns
        -------
        Tensor
            Tensor of zeros with ones at positions where ``lags == 0``.
        """
        # Ensure the output has the same dtype/device as the input.
        return torch.exp(-self.lambda_ * torch.abs(lags))

class OUProcessFDD(StationaryProcessFDD):
    """
    Finite‑dimensional distribution for a :class:`NoiseProcess`.

    Because the variables are independent the joint cumulant is the sum of the
    marginal cumulants.  Sampling (if a ``sample_func`` was given) simply draws
    independent copies for each time point.
    """

    def __init__(
        self,
        process: OUProcess,
        times: Tensor,
        *,
        rng: Optional[torch.Generator] = None,
        arg_check: bool = True,
        theta_batch_first: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        process :
            Parent :class:`NoiseProcess` instance.
        times :
            Strictly increasing 1‑D tensor of observation times.
        rng :
            Optional generator used by the (optional) sampler.
        arg_check :
            Whether to perform validation of ``theta``/``times`` in the
            ``cumulant`` method.
        theta_batch_first :
            Default orientation for ``theta`` (overwrites the global default
            stored on the parent process).
        """
        self.times = times
        self.process = process
        self.rng = rng

        # Validation settings – they shadow the global defaults only for this
        # particular FDD instance.
        self.arg_check = arg_check
        self.theta_batch_first_default = theta_batch_first

    # -----------------------------------------------------------------
    # Public API required by ``StationaryProcessFDD``.
    # -----------------------------------------------------------------
    def cumulant(
        self,
        theta: Tensor,
        theta_batch_first: Optional[bool] = None,
    ) -> Tensor:
        """
        Joint cumulant κ(θ) for the vector ``(Y_{t₁}, …, Y_{t_D})``.

        Because the variables are independent, the joint cumulant equals the
        sum of the *marginal* cumulants evaluated at each coordinate of ``θ``:

        .. math::
            κ(θ) = \\sum_{d=1}^{D} ℓ\\bigl(θ_d\\bigr),

        where ``ℓ`` is the marginal cumulant supplied at construction time.

        Parameters
        ----------
        theta :
            Tensor of Fourier arguments.  Accepted layouts are ``(B, D)`` or
            ``(D, B)`` depending on ``theta_batch_first``.
        theta_batch_first :
            Overrides the default orientation stored on the parent process.

        Returns
        -------
        Tensor
            Cumulant values of shape ``(B,)`` (one value per batch).
        """
        # -----------------------------------------------------------------
        # Resolve orientation and run optional validation.
        # -----------------------------------------------------------------
        if theta_batch_first is None:
            theta_batch_first = self.theta_batch_first_default

        if self.arg_check:
            self.process._validate_cumulant_args(theta, self.times, theta_batch_first)

        # Normalise to internal (B, D) layout.
        theta_norm = (
            theta if theta_batch_first else theta.swapaxes(0, 1)
        )  # shape (B, D)
        
        cov_matrix = self.process.acf(torch.abs(self.times - self.times.t())) * self.process.sigma**2 / (2 * self.process.lambda_)  
        dirac_comp = 1.0j * self.process.mu * theta_norm.sum(dim = 1)
        gauss_comp = torch.einsum('bj,jk,bk->b', theta_norm, cov_matrix, theta_norm)
        return dirac_comp + gauss_comp

    def sample(
        self,
        batch_size: int = 1,
        batch_first: bool = True,
        unsqueeze_last: bool = True,
        dt: float = 0.05
    ) -> Tensor:
        """
        Draw independent trajectories for the supplied grid.

        The method uses the optional ``sample_func`` that the user may have
        provided when constructing the parent :class:`NoiseProcess`.  If no
        sampler was supplied a ``NotImplementedError`` is raised.

        Parameters
        ----------
        batch_size :
            Number of independent copies to generate.
        batch_first :
            If ``True`` (default) the output has shape
            ``(batch_size, D, 1)``; otherwise the shape is
            ``(D, batch_size, 1)``.
        unsqueeze_last :
            If ``True`` (default) a trailing singleton dimension is kept,
            matching the original library’s behaviour.

        Returns
        -------
        Tensor
            Sampled trajectories of shape ``(batch_size, D, 1)`` (or transposed
            when ``batch_first=False``).
        """

        base_grid_steps = int((self.times[-1] / dt).item()) + 1
        base_grid_time = base_grid_steps * dt
        base_grid = torch.linspace(0, base_grid_time, base_grid_steps) 
        
        Time = torch.unique(torch.cat((base_grid, self.times)))
        total_steps = Time.shape[0]
        sample_idxs = torch.searchsorted(Time, self.times)

        Traj = torch.zeros((batch_size, total_steps))
        
        Traj[:, 0] = torch.normal(self.process.mu.view(batch_size), self.process.sigma*torch.sqrt(torch.tensor(1 / (2 * self.process.lambda_))).view(batch_size), generator = self.rng)
        # Shape handling required by the public API.
        
        dTime = Time[1:] - Time[:-1] 
        for i in range(1, total_steps):
            dW_t = torch.normal(torch.tensor(0.0).view(batch_size), torch.sqrt(torch.tensor(dTime[i - 1])).view(batch_size), generator=self.rng)
            Traj[:, i] = Traj[:, i-1] + self.process.lambda_ * (self.process.mu -  Traj[:, i-1]) * dTime[i - 1] + self.process.sigma * dW_t
        if not batch_first:
            Traj = Traj.T
        if unsqueeze_last:
            Traj = Traj.unsqueeze(-1)
        return Traj[:, sample_idxs]


