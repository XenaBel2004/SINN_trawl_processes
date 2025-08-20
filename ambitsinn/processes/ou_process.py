#!/usr/bin/env python
# -*- coding: ascii -*-
"""Implementation of Ornstein-Uhlenbeck (OU) process."""

from __future__ import annotations

from typing import Optional, cast

import torch
from torch import Generator, Tensor
from torch.distributions.normal import Normal

from .base import StationaryProcessFDD, StationaryStochasticProcess


class OUProcess(StationaryStochasticProcess):
    r"""Ornstein-Uhlenbeck (OU) process.

    The OU process is the stationary solution for the stochastic
    differential equation

    .. math::
        dX_t = \lambda (\mu - X_t)\,dt + \sigma\,dW_t,

    where :math:`\lambda > 0` is the speed of mean reversion,
    :math:`\mu` the long-term mean, :math:`\sigma > 0` the noise
    intensity, and :math:`W_t` a standard Wiener process.

    It's marginal distribution is Gaussian with
    mean :math:`\mu` and variance :math:`\frac{\sigma^2}{(2\lambda)}`.
    Its autocovariance function is

    .. math::
        \operatorname{Cov}(X_t, X_{t+s}) = \frac{\sigma^2}{2\lambda}
        \exp\bigl(-\lambda |s|\bigr).

    Parameters
    ----------
    mu
        Long-term mean :math:`\mu`.
    sigma
        Noise strength :math:`\sigma`.
    lambda\_
        Speed of mean reversion :math:`\lambda`.
    **default_opts
        Additional keyword arguments forwarded to the base class
        (e.g., ``device`` to select the torch device).

    Notes
    -----
    https://encyclopediaofmath.org/wiki/Ornstein-Uhlenbeck_process

    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        lambda_: float = 1.0,
        **default_opts,
    ) -> None:
        r"""Create a new OU process instance.

        The parameters are stored as tensors on the optional device
        provided in ``default_opts``.  The base class handles generic
        functionality for stationary stochastic processes.

        Parameters
        ----------
        mu
            Long-term mean.
        sigma
            Noise strength.
        lambda\_
            Mean-reversion speed.
        **default_opts:
            Additional options forwarded to :class:`StationaryStochasticProcess`.

        """
        self.lambda_: Tensor = torch.tensor(lambda_, device=default_opts.get("device"))
        self.sigma: Tensor = torch.tensor(sigma, device=default_opts.get("device"))
        self.mu: Tensor = torch.tensor(mu, device=default_opts.get("device"))

        super().__init__(**default_opts)

    def at_times(self, times: Tensor, rng: Optional[Generator] = None) -> OUProcessFDD:
        """Create a finite-dimensional distribution (FDD) for a time grid.

        Parameters
        ----------
        times
            Strictly increasing 1-D tensor of observation times.
         rng
            Random number generator used by the sampler.

        Returns
        -------
        OUProcessFDD
            Corresponding finite-dimensional distribution object.

        """

        return OUProcessFDD(self, times, rng=rng)

    def pdf(self, x: Tensor) -> Tensor:
        r"""Evaluate the marginal probability density function.

        At any fixed time the OU process is normally distributed with mean
        :math:`\mu` and variance :math:`\frac{sigma^2}(2\lambda)`.  The density is

        .. math::
            f_X(x) = \frac{1}{\sqrt{2\pi\,\sigma^2/(2\lambda)}}
            \exp\Bigl(-\lambda \frac{(x-\mu)^2}{\sigma^2}\Bigr).

        Parameters
        ----------
        x
            Points at which to evaluate the density.  Output shape matches
            the shape of ``x``.

        Returns
        -------
        Tensor
            Probability density values.

        """
        return Normal(self.mu, self.sigma / torch.sqrt(2 * self.lambda_)).log_prob(x).exp()

    def acf(self, lags: Tensor) -> Tensor:
        r"""Autocorrelation function of the stationary OU process.

        For a lag :math:`\tau` the autocorrelation is

        .. math::
            \rho(\tau) = \exp\bigl(-\lambda\,|\tau|\bigr).

        Parameters
        ----------
        lags
            Non-negative lag values.

        Returns
        -------
        Tensor
            Autocorrelation values with the same shape as ``lags``.

        """
        return torch.exp(-self.lambda_ * torch.abs(lags))


class OUProcessFDD(StationaryProcessFDD):
    """Finite-dimensional distribution of an OU process on a given grid.

    The class provides the cumulant generating function and a sampler that
    draws paths at the requested observation times.

    Parameters
    ----------
    process
        The underlying Ornstein-Uhlenbeck process.
    times
        Strictly increasing 1-D tensor of observation times.
    rng
        Random number generator used by the sampler.

    """

    def __init__(self, process: OUProcess, times: Tensor, rng: Optional[torch.Generator] = None) -> None:
        """Create the FDD for ``process`` at the supplied ``times``."""
        super().__init__(times, process, rng=rng)
        self.process: OUProcess = cast(OUProcess, self.process)  # MyPy fix

    def cumulant(
        self,
        theta: Tensor,
        theta_batch_first: Optional[bool] = None,
    ) -> Tensor:
        r"""Cumulant of the joint distribution.

        For a Gaussian vector :math:`X` with mean :math:`mu` and covariance matrix
        :math:`\Sigma` the cumulant is given by is

        .. math::
            \kappa(\theta) = \mathrm{i}\,\mu\,\mathbf{1}^\top\theta
            - \frac{1}{2}\,\theta^\top\Sigma\,\theta,

        where ``theta`` is a (possibly batched) vector of frequencies.
        This method evaluates the formula for the distribution of the OU process
        at the observation times stored in ``self.times``.

        Parameters
        ----------
        theta
            Frequency vector(s).  Shape ``(batch, dim)`` if
            ``theta_batch_first`` is ``True`` or ``(dim, batch)`` otherwise.
        theta_batch_first
            If ``True``, interpret the first dimension of ``theta`` as the
            batch dimension.  If ``None`` the default behaviour of the base
            class is used.

        Returns
        -------
        Tensor
            Cumulant evaluated at each ``theta`` in the batch.

        """

        if self.process.arg_check:
            self.process._validate_cumulant_args(theta, self.times, theta_batch_first)
        theta_norm = self.process._normalize_theta(theta, theta_batch_first)

        cov_matrix = (
            self.process.acf(torch.abs(self.times.view(-1, 1) - self.times.view(-1, 1).t()))
            * self.process.sigma**2
            / (2 * self.process.lambda_)
        )
        dirac_comp = 1.0j * self.process.mu * theta_norm.sum(dim=1)
        gauss_comp = torch.einsum("bj,jk,bk->b", theta_norm, cov_matrix, theta_norm)
        return dirac_comp - gauss_comp / 2

    def sample(
        self,
        batch_size: int = 1,
        batch_first: Optional[bool] = None,
        unsqueeze_last: Optional[bool] = None,
        dt: float = 0.05,
    ) -> Tensor:
        """Draw sample paths of the OU process on the observation grid.

        The method uses an Euler-Maruyama discretisation on a fine auxiliary
        grid (step size ``dt``) and extracts the values at the requested
        observation times.  The returned tensor can be reshaped using
        ``batch_first`` and ``unsqueeze_last`` to match the conventions of the
        base class.

        Parameters
        ----------
        batch_size
            Number of independent sample paths to generate.
        batch_first
            If ``True``, the batch dimension will be the first axis of the
            output; if ``False`` it will be the last axis.  If ``None`` the
            default of the base class is used.
        unsqueeze_last
            If ``True``, a singleton dimension is appended to the output.
        dt : float, optional (default=0.05)
            Time step for the Euler-Maruyama integration.

        Returns
        -------
        Tensor
            Sampled values at ``self.times``.  Shape depends on the
            ``batch_first`` and ``unsqueeze_last`` arguments.

        """
        base_grid_steps: int = int((self.times[-1] / dt).item()) + 1
        base_grid_time: float = base_grid_steps * dt
        base_grid = torch.linspace(0.0, base_grid_time, base_grid_steps, device=self.process.device)

        # Merge fine grid with the observation times.
        Time = torch.unique(torch.cat((base_grid, self.times)))
        total_steps = Time.shape[0]
        sample_idxs = torch.searchsorted(Time, self.times)

        # Allocate trajectory tensor.
        Traj = torch.zeros((batch_size, total_steps), device=self.process.device)

        # Initial value drawn from the stationary distribution.
        Traj[:, 0] = torch.normal(
            self.process.mu.broadcast_to(batch_size),
            self.process.sigma * torch.sqrt(1 / (2 * self.process.lambda_)).broadcast_to(batch_size),
            generator=self.rng,
        )
        # Time increments.
        dTime = Time[1:] - Time[:-1]
        for i in range(1, total_steps):
            # Wiener increment.
            dW_t = torch.normal(
                torch.tensor(0.0, device=self.process.device).broadcast_to(batch_size),
                torch.sqrt(dTime[i - 1]).broadcast_to(batch_size),
                generator=self.rng,
            )
            # Euler-Maruyama update.
            Traj[:, i] = (
                Traj[:, i - 1]
                + self.process.lambda_ * (self.process.mu - Traj[:, i - 1]) * dTime[i - 1]
                + self.process.sigma * dW_t
            )

        # Return only the requested observation times, reshaped as requested.
        return self.process._normalize_sample(
            Traj[:, sample_idxs],
            sample_batch_first=batch_first,
            sample_unsqueeze=unsqueeze_last,
        )


__all__ = [
    "OUProcess",
    "OUProcessFDD",
]
