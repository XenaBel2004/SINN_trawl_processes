# -*- coding: ascii -*-
"""
NoiseProcess - a collection of *i.i.d.* random variables.

The process is defined solely by the log-characteristic function of a single
margin (the *cumulant function*) and, torch Distribution object.  Because the variables are independent, the joint cumulant is simply
the sum of the marginal cumulants and the autocorrelation is 1 at lag 0 and
0 elsewhere.

The implementation conforms to the abstract API defined in ``base.py``:
* ``NoiseProcess`` derives from :class:`StationaryStochasticProcess`.
* ``NoiseFDD`` derives from :class:`StationaryProcessFDD`.
"""

from __future__ import annotations

from typing import Callable, Optional, cast

import torch
from torch import Tensor
from torch.distributions.distribution import Distribution

from .base import StationaryStochasticProcess, StationaryProcessFDD


class NoiseProcess(StationaryStochasticProcess):
    """
    Stationary *independent-noise* process  ``Y_t``.

    For every time point ``t`` the random variable ``Y_t`` has the same
    marginal law, characterised by its **log-characteristic function**

    .. math::
        \\kappa(\\theta) = \\log\\mathbb{E}\\bigl[\\exp{\\mathrm{i}\\,\\tehta Y_0}\\bigr].

    The user supplies that function via ``cumulant_func``.  Optionally a
    callable that draws i.i.d. samples from the marginal law can be provided
    (``sample_func``) and/or a PDF implementation (``pdf_func``).

    Parameters
    ----------
    cumulant_func :
        Callable :math: `\\theta \to \\kappa(\\theta)`.  ``theta`` is a real-valued ``torch.Tensor`` and the
        return must be a complex-valued tensor of the same shape.
    distr :
        torch.distributions.Distribution instance. Used for computing pdf and sampling
        should be univariate
    theta_batch_first :
        Default orientation for ``theta`` (see :class:`StationaryStochasticProcess`).
    arg_check :
        Whether to perform run-time validation of arguments.
    """

    def __init__(
        self,
        distr: Distribution,
        cumulant_func: Callable[[Tensor], Tensor],
        **default_opts,
    ) -> None:
        super().__init__(**default_opts)

        self.distr: Distribution = distr
        self.cumulant_func: Callable[[Tensor], Tensor] = cumulant_func

    def at_times(
        self, times: Tensor, rng: Optional[torch.Generator] = None
    ) -> "NoiseFDD":
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
        return NoiseFDD(self, times, rng=rng)

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
        return self.distr.log_prob(x).exp()

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
        zeros: Tensor = torch.zeros_like(lags, dtype=torch.float64)
        ones: Tensor = torch.ones_like(lags, dtype=torch.float64)
        return torch.where(lags == 0, ones, zeros)


class GaussianNoise(NoiseProcess):
    def __init__(self, mean: float = 0.0, var: float = 1.0, **default_opts) -> None:
        self.mean: Tensor = torch.tensor(mean, device=default_opts.get("device"))
        self.var: Tensor = torch.tensor(var, device=default_opts.get("device"))
        distr: Distribution = torch.distributions.normal.Normal(
            self.mean, self.var**0.5
        )
        def cumulant(u: Tensor) -> Tensor:
            return torch.exp(
                    1.0j * self.mean * u - self.var * u**2 / 2
                )
        super().__init__(distr, cumulant, **default_opts)


class ExponentialNoise(NoiseProcess):
    def __init__(self, rate: float = 1.0, **default_opts) -> None:
        self.rate: Tensor = torch.tensor(rate, device=default_opts.get("device"))
        distr: Distribution = torch.distributions.gamma.Gamma(
            torch.tensor(1.0), self.rate
        )
        def cumulant(u: Tensor) -> Tensor:
            return -torch.log(
                    1 - 1.0j * u / self.rate
                )
        super().__init__(distr, cumulant, **default_opts)


class NoiseFDD(StationaryProcessFDD):
    """
    Finite-dimensional distribution for a :class:`NoiseProcess`.

    Because the variables are independent the joint cumulant is the sum of the
    marginal cumulants.  Sampling (if a ``sample_func`` was given) simply draws
    independent copies for each time point.
    """

    def __init__(
        self,
        process: NoiseProcess,
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
            Strictly increasing 1-D tensor of observation times.
        rng :
            Optional generator used by the (optional) sampler.
        arg_check :
            Whether to perform validation of ``theta``/``times`` in the
            ``cumulant`` method.
        theta_batch_first :
            Default orientation for ``theta`` (overwrites the global default
            stored on the parent process).
        """
        super().__init__(times, process, rng=rng)
        self.process: NoiseProcess = cast(NoiseProcess, self.process)  # MyPy fix

    # -----------------------------------------------------------------
    # Public API required by ``StationaryProcessFDD``.
    # -----------------------------------------------------------------
    def cumulant(
        self,
        theta: Tensor,
        theta_batch_first: Optional[bool] = None,
    ) -> Tensor:
        """
        Joint cumulant :math: `\\kappa(\\theta)` for the vector :math: `(Y_{t_1}, ..., Y_{t_D})`.

        Because the variables are independent, the joint cumulant equals the
        sum of the *marginal* cumulants evaluated at each coordinate of :math: `\\theta`:

        .. math::
            \\kappa(\\theta) = \\sum_{d=1}^{D} \\ell\\bigl(\\theta_d\\bigr),

        where ``\\ell`` is the marginal cumulant supplied at construction time.

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
        if self.process.arg_check:
            self.process._validate_cumulant_args(theta, self.times)

        # Normalise theta to ``(B, D)`` layout.
        theta_norm = self.process._normalize_theta(theta, theta_batch_first)

        # Apply the marginal cumulant element-wise and sum across the time axis.
        # ``_cumulant_func`` returns a complex tensor of the same shape as the
        # input; ``torch.sum(..., dim=1)`` collapses the D dimension.
        marginal_cumulants = self.process.cumulant_func(theta_norm)
        return torch.sum(marginal_cumulants, dim=1)

    def sample(
        self,
        batch_size: int = 1,
        batch_first: Optional[bool] = None,
        unsqueeze_last: Optional[bool] = None,
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
            matching the original library's behaviour.

        Returns
        -------
        Tensor
            Sampled trajectories of shape ``(batch_size, D, 1)`` (or transposed
            when ``batch_first=False``).
        """
        samples = self.process.distr.sample(
            sample_shape=(batch_size, self.times.shape[0])
        )

        # Shape handling required by the public API.
        return self.process._normalize_sample(
            samples, sample_batch_first=batch_first, sample_unsqueeze=unsqueeze_last
        )
