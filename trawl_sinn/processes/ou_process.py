# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, cast

import torch
from torch import Tensor
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

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        lambda_: float = 1.0,
        **default_opts,
    ) -> None:
        self.lambda_: Tensor = torch.tensor(lambda_, device=default_opts.get("device"))
        self.sigma: Tensor = torch.tensor(sigma, device=default_opts.get("device"))
        self.mu: Tensor = torch.tensor(mu, device=default_opts.get("device"))

        super().__init__(**default_opts)

    def at_times(
        self, times: Tensor, rng: Optional[torch.Generator] = None
    ) -> "OUProcessFDD":
        """
        Create a finite-dimensional distribution (FDD) for the supplied
        observation grid.

        The returned ``NoiseFDD`` knows the grid, the cumulant function,
        and (if supplied) the sampler / PDF.

        Parameters
        ----------
        times :
            Strictly increasing 1-D tensor of observation times.
        rng :
            Optional ``torch.Generator`` that will be passed to the sampler
            (if it accepts a ``generator`` argument).

        Returns
        -------
        NoiseFDD
            An FDD object that implements :meth:`cumulant`, :meth:`charfunc`,
            and (optionally) :meth:`sample`.
        """
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
        return (
            torch.distributions.normal.Normal(
                self.mu, self.sigma / torch.sqrt(2 * self.lambda_)
            )
            .log_prob(x)
            .exp()
        )

    def acf(self, lags: Tensor) -> Tensor:
        """
        Autocorrelation function for an i.i.d. process.

        By definition the autocorrelation equals ``1`` at lag ``0`` and ``0``
        for any non-zero lag.

        Parameters
        ----------
        lags :
            Tensor of non-negative lags (same shape as the return value).

        Returns
        -------
        Tensor
            Tensor of zeros with ones at positions where ``lags == 0``.
        """
        # Ensure the output has the same dtype/device as the input.
        return torch.exp(-self.lambda_ * torch.abs(lags))


class OUProcessFDD(StationaryProcessFDD):
    def __init__(
        self, process: OUProcess, times: Tensor, rng: Optional[torch.Generator] = None
    ) -> None:
        super().__init__(times, process, rng=rng)
        self.process: OUProcess = cast(OUProcess, self.process)  # MyPy fix

    # -----------------------------------------------------------------
    # Public API required by ``StationaryProcessFDD``.
    # -----------------------------------------------------------------
    def cumulant(
        self,
        theta: Tensor,
        theta_batch_first: Optional[bool] = None,
    ) -> Tensor:
        # -----------------------------------------------------------------
        # Resolve orientation and run optional validation.
        # -----------------------------------------------------------------
        if self.process.arg_check:
            self.process._validate_cumulant_args(theta, self.times, theta_batch_first)

        # Normalise to internal (B, D) layout.
        theta_norm = self.process._normalize_theta(theta, theta_batch_first)
        cov_matrix = (
            self.process.acf(torch.abs(self.times - self.times.t()))
            * self.process.sigma**2
            / (2 * self.process.lambda_)
        )
        dirac_comp = 1.0j * self.process.mu * theta_norm.sum(dim=1)
        gauss_comp = torch.einsum("bj,jk,bk->b", theta_norm, cov_matrix, theta_norm)
        return dirac_comp + gauss_comp

    def sample(
        self,
        batch_size: int = 1,
        batch_first: Optional[bool] = None,
        unsqueeze_last: Optional[bool] = None,
        dt: float = 0.05,
    ) -> Tensor:
        base_grid_steps: int = int((self.times[-1] / dt).item()) + 1
        base_grid_time: float = base_grid_steps * dt
        base_grid = torch.linspace(
            0.0, base_grid_time, base_grid_steps, device=self.process.device
        )

        Time = torch.unique(torch.cat((base_grid, self.times)))
        total_steps = Time.shape[0]
        sample_idxs = torch.searchsorted(Time, self.times)

        Traj = torch.zeros((batch_size, total_steps))

        Traj[:, 0] = torch.normal(
            self.process.mu.broadcast_to(batch_size),
            self.process.sigma
            * torch.sqrt(1 / (2 * self.process.lambda_)).broadcast_to(batch_size),
            generator=self.rng,
        )
        # Shape handling required by the public API.

        dTime = Time[1:] - Time[:-1]
        for i in range(1, total_steps):
            dW_t = torch.normal(
                torch.tensor(0.0).broadcast_to(batch_size),
                torch.sqrt(dTime[i - 1]).broadcast_to(batch_size),
                generator=self.rng,
            )
            Traj[:, i] = (
                Traj[:, i - 1]
                + self.process.lambda_
                * (self.process.mu - Traj[:, i - 1])
                * dTime[i - 1]
                + self.process.sigma * dW_t
            )
        return self.process._normalize_sample(
            Traj[:, sample_idxs],
            sample_batch_first=batch_first,
            sample_unsqueeze=unsqueeze_last,
        )
