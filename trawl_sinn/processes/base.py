# base.py
"""
Base classes that define the public API for stationary stochastic processes
and their finite‑dimensional distributions (FDDs).

The module provides:
* ``StationaryStochasticProcess`` – an abstract process class with validation,
  cumulant/characteristic‑function helpers and a generic sampling entry point.
* ``StationaryProcessFDD`` – an abstract FDD class that knows the observation
  grid and implements the heavy‑lifting for cumulants, characteristic functions
  and sampling.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Callable, Optional
from abc import ABC, abstractmethod


class StationaryStochasticProcess(ABC):
    """
    Abstract base class for *stationary* stochastic processes.

    The class supplies common utilities (argument validation, orientation
    handling, cumulant/characteristic‑function evaluation, sampling) that
    concrete subclasses can reuse.

    Parameters
    ----------
    arg_check:
        Whether to perform run‑time validation of user supplied arguments.
        Turning this off can give a modest speed‑up when you are certain inputs
        are well‑formed.
    theta_batch_first:
        Default assumption for the orientation of a user supplied ``theta``.
        If ``True`` the first dimension corresponds to the batch dimension, i.e.
        ``theta`` has shape ``(B, D)`` where ``D`` is the number of observation
        times.  If ``False`` the shape is ``(D, B)`` and the class will
        internally transpose the tensor.
    """

    def __init__(
        self,
        *,
        arg_check: bool = True,
        theta_batch_first: bool = True,
        sample_batch_first: bool = True,
        sample_unsqueeze: bool = True,
        device: Optional[torch.Device] = None,
    ) -> None:
        self.device: torch.Device = device
        self.arg_check: bool = arg_check
        self.sample_unsqueeze_default: bool = sample_unsqueeze
        self.theta_batch_first_default: bool = theta_batch_first
        self.sample_batch_first_default: bool = sample_batch_first

    # -----------------------------------------------------------------
    # validation helpers
    # -----------------------------------------------------------------
    def _validate_times(self, times: Tensor) -> None:
        """
        Validate a tensor of observation times.

        Checks performed
        ----------------
        * ``times`` is one‑dimensional.
        * No element is ``NaN``.
        * Strictly increasing (no equal neighbours).
        * All entries are unique (duplicates raise an error).

        Raises
        ------
        ValueError
            If any of the above conditions is violated.
        """
        if times.ndim != 1:
            raise ValueError("`times` must be a one‑dimensional torch.Tensor")
        if torch.any(torch.isnan(times)):
            raise ValueError("`times` must not contain NaN")
        if not torch.all(times[1:] > times[:-1]):
            raise ValueError("`times` must be strictly increasing")
        if times.numel() != torch.unique(times).numel():
            raise ValueError("`times` must contain unique entries")
        if (self.device is not None) and times.device != self.device:
            raise ValueError(
                f"`times.device` is {times.device}, should be {self.device}"
            )

    def _validate_theta(self, theta: Tensor) -> None:
        """
        Validate a tensor of ``theta`` arguments used for cumulants/characteristic
        functions.

        Checks performed
        ----------------
        * ``theta`` is two‑dimensional.
        * No element is ``NaN``.

        Raises
        ------
        ValueError
            If either condition fails.
        """
        if theta.ndim != 2:
            raise ValueError("`theta` must be a two‑dimensional torch.Tensor")
        if torch.any(torch.isnan(theta)):
            raise ValueError("`theta` must not contain NaN")
        if (self.device is not None) and theta.device != self.device:
            raise ValueError(
                f"`theta.device` is {theta.device}, should be {self.device}"
            )

    def _validate_cumulant_args(
        self,
        theta: Tensor,
        times: Tensor,
        theta_batch_first: bool,
    ) -> None:
        """
        Validate arguments passed to :meth:`cumulant` / :meth:`charfunc`.

        The checks are a superset of those performed by ``_validate_theta`` and
        ``_validate_times`` and additionally enforce shape compatibility between
        ``theta`` and ``times`` according to the orientation flag.

        Raises
        ------
        ValueError
            If a shape or device incompatibility is detected.
        """
        self._validate_theta(theta)
        self._validate_times(times)

        if theta_batch_first:
            # (B, D) layout – ``D`` must match the number of observation times.
            if theta.shape[1] != times.shape[0]:
                raise ValueError(
                    "`theta.shape[1]` must equal `len(times)` when `theta_batch_first=True`"
                )
        else:
            # (D, B) layout – ``D`` must match the number of observation times.
            if theta.shape[0] != times.shape[0]:
                raise ValueError(
                    "`theta.shape[0]` must equal `len(times)` when `theta_batch_first=False`"
                )

    def _normalize_theta(
        self, theta: Tensor, theta_batch_first: Optional[bool] = None
    ) -> Tensor:
        """
        Normalise ``theta`` to the internal ``(B, D)`` layout.

        Parameters
        ----------
        theta
            Input tensor of shape either ``(B, D)`` (if ``theta_batch_first`` is
            ``True``) or ``(D, B)`` (if ``False``).
        theta_batch_first
            Overrides the default orientation stored on the process instance.
            If ``None`` the default value from ``self.theta_batch_first_default``
            is used.

        Returns
        -------
        Tensor
            ``theta`` in ``(B, D)`` layout.
        """
        if theta_batch_first is None:
            theta_batch_first = self.theta_batch_first_default
        if not theta_batch_first:
            theta = theta.swapaxes(0, 1)
        return theta

    def _normalize_sample(
        self,
        data: Tensor,
        sample_unsqueeze: Optional[bool] = None,
        sample_batch_first: Optional[bool] = None,
    ) -> Tensor:
        if sample_unsqueeze is None:
            sample_unsqueeze = self.sample_unsqueeze_default
        if sample_batch_first is None:
            sample_batch_fist = self.sample_batch_first_default
        if not sample_batch_first:
            data = data.T
        if sample_unsqueeze:
            data = data.unsqueeze(-1)
        return data

    # -----------------------------------------------------------------
    # public API (delegates to a finite‑dimensional‑distribution object)
    # -----------------------------------------------------------------
    @abstractmethod
    def at_times(
        self, times: Tensor, rng: Optional[torch.Generator] = None
    ) -> "StationaryProcessFDD":
        """
        Build a *finite‑dimensional distribution* (FDD) object for the
        supplied observation grid ``times``.

        Sub‑classes must return an instance of a concrete subclass of
        :class:`StationaryProcessFDD` that implements the heavy lifting for
        cumulants, characteristic functions and sampling.

        Parameters
        ----------
        times
            A 1‑D strictly increasing tensor of observation times.
        rng
            Optional ``torch.Generator`` for reproducible sampling.

        Returns
        -------
        StationaryProcessFDD
            An object that knows the grid and the parent process.
        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self, x: Tensor) -> Tensor:
        """
        Marginal probability density function of the process.

        Sub‑classes must provide a vector‑valued (or scalar) density
        evaluated at the points ``x``.  The return shape should broadcast
        with ``x``.

        Parameters
        ----------
        x
            Points at which to evaluate the density.

        Returns
        -------
        Tensor
            Density values with the same dtype as ``x``.
        """
        raise NotImplementedError

    @abstractmethod
    def acf(self, lags: Tensor) -> Tensor:
        """
        Autocorrelation function (ACF) for the process.

        The implementation should return a tensor of autocorrelations for each
        lag in ``lags``.  The shape must broadcast with ``lags``.

        Parameters
        ----------
        lags
            Non‑negative lags (time differences) for which to compute the ACF.

        Returns
        -------
        Tensor
            Autocorrelation values.
        """
        raise NotImplementedError

    def cumulant(
        self,
        theta: Tensor,
        times: Tensor,
        theta_batch_first: Optional[bool] = None,
    ) -> Tensor:
        """
        Joint cumulant ``κ(θ) = log E[exp(i <θ, X[times]>)]`` for the supplied
        observation grid.

        The function forwards the call to the corresponding :class:`StationaryProcessFDD`
        instance created by :meth:`at_times`.

        Parameters
        ----------
        theta
            Tensor of Fourier arguments.  Accepted layouts are ``(B, D)`` or
            ``(D, B)`` depending on ``theta_batch_first``.
        times
            1‑D tensor of observation times.
        theta_batch_first
            Overrides the class‑level default orientation.

        Returns
        -------
        Tensor
            Tensor of shape ``(B,)`` containing the cumulant for each batch.
        """
        if theta_batch_first is None:
            theta_batch_first = self.theta_batch_first_default
        return self.at_times(times).cumulant(theta, theta_batch_first)

    def charfunc(
        self,
        theta: Tensor,
        times: Tensor,
        theta_batch_first: Optional[bool] = None,
    ) -> Tensor:
        """
        Joint characteristic function ``φ(θ) = exp(κ(θ))``.
        """
        return torch.exp(self.cumulant(theta, times, theta_batch_first))

    # -----------------------------------------------------------------
    # generic sampling entry‑point (uses the underlying FDD)
    # -----------------------------------------------------------------
    def sample(
        self,
        times: Tensor,
        *,
        rng: Optional[torch.Generator] = None,
        batch_size: int = 1,
        batch_first: bool = True,
        unsqueeze_last: bool = True,
        **kwargs,
    ) -> Tensor:
        """
        Sample trajectories of the process on a user supplied grid.

        Parameters
        ----------
        times
            1‑D increasing tensor of observation times.
        rng
            Optional ``torch.Generator`` to obtain deterministic draws.
        batch_size
            Number of independent trajectories to return.
        batch_first
            If ``True`` (default) the output tensor has shape
            ``(batch_size, len(times), …)``.  If ``False`` the batch dimension
            is placed last.
        unsqueeze_last
            If ``True`` an additional singleton dimension is appended at the
            end (matching the original library convention).

        Returns
        -------
        Tensor
            Sampled trajectories.  When ``unsqueeze_last=True`` the shape is
            ``(batch_size, len(times), 1)`` (or ``(len(times), batch_size, 1)``
            when ``batch_first=False``).  When ``unsqueeze_last=False`` the final
            singleton dimension is omitted.
        """
        return self.at_times(times, rng=rng).sample(
            batch_size=batch_size,
            batch_first=batch_first,
            unsqueeze_last=unsqueeze_last,
            **kwargs,
        )


class StationaryProcessFDD(ABC):
    """
    Abstract helper that knows the observation grid and the parent process.

    All computationally heavy operations – cumulant/characteristic‑function
    evaluation and sampling – are implemented by concrete subclasses.
    """

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------
    @abstractmethod
    def cumulant(
        self,
        theta: Tensor,
        theta_batch_first: bool = True,
    ) -> Tensor:
        """
        Joint cumulant ``κ(θ)`` for the vector
        ``(X_{t₁}, …, X_{t_D})``.

        The return tensor must have shape ``(B,)``, where ``B`` is the batch
        dimension supplied with ``theta`` (or implicitly ``1`` if no batch was
        provided).

        Parameters
        ----------
        theta
            Tensor of Fourier arguments; accepted layout is ``(B, D)`` or
            ``(D, B)`` depending on ``theta_batch_first``.
        theta_batch_first
            Overrides the default orientation stored in the parent process.

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
        """Characteristic function ``φ(θ) = exp(κ(θ))``."""
        return torch.exp(self.cumulant(theta, theta_batch_first))

    @abstractmethod
    def sample(
        self,
        batch_size: int = 1,
        batch_first: bool = True,
        unsqueeze_last: bool = True,
        **kwargs,
    ) -> Tensor:
        """
        Draw trajectories for the current grid.

        Parameters
        ----------
        batch_size
            Number of independent copies to generate.
        batch_first
            If ``True`` (default) the batch dimension appears first in the
            returned tensor.  If ``False`` the batch dimension is moved to the
            last axis.
        unsqueeze_last
            If ``True`` (default) a singleton dimension is appended at the end,
            matching the original library convention.

        Returns
        -------
        Tensor
            Sampled trajectories, of shape ``(batch_size, D, 1)`` (or
            ``(D, batch_size, 1)`` when ``batch_first=False``) unless
            ``unsqueeze_last=False``.
        """
        raise NotImplementedError
