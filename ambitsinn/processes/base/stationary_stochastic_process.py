# -*- coding: ascii -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
from torch import Generator, Tensor

if TYPE_CHECKING:  # pragma: no cover
    from .stationary_process_fdd import StationaryProcessFDD


class StationaryStochasticProcess(ABC):
    """Abstract base class for *stationary* stochastic processes.

    This class supplies a common toolbox (argument validation, orientation
    handling, cumulant/characteristic-function evaluation and sampling) that
    concrete subclasses can reuse.  All heavy-weight operations are delegated
    to a :class:`StationaryProcessFDD` instance built by :meth:`at_times`.
    """

    def __init__(
        self,
        *,
        arg_check: bool = True,
        theta_batch_first: bool = True,
        sample_batch_first: bool = False,
        sample_unsqueeze: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """Construct a :class:`StationaryStochasticProcess`.

        Parameters
        ----------
        arg_check : bool, optional
            Whether to validate user-supplied arguments (default: ``True``).
            Disabling validation can give a modest speed-up when you are
            certain inputs are well-formed.
        theta_batch_first : bool, optional
            Default orientation for ``theta`` tensors passed to :meth:`cumulant`
            and :meth:`charfunc`.  ``True`` means the batch dimension comes
            first (shape ``(B, D)``); ``False`` swaps the axes (shape
            ``(D, B)``).  (default: ``True``).
        sample_batch_first : bool, optional
            Default orientation for the tensors returned by :meth:`sample`.  If
            ``True`` the batch dimension appears first
            (``(B, D, ...)``); if ``False`` the batch dimension is placed
            last (``(D, B, ...)``).  (default: ``True``).
        sample_unsqueeze : bool, optional
            Whether to append a trailing singleton dimension to sampled
            trajectories. If ``True``, sampled trajectories will have shape
            either (B,D,1) or (D,B,1) (default: ``True``).
        device : torch.device or ``None``, optional
            If given, all tensors supplied to the process must live on this
            device.  The validation helpers raise ``ValueError`` if a tensor is
            on a different device.

        Notes
        -----
        The arguments are stored as instance attributes with the suffix
        ``_default`` so that they can be overridden per-call.

        """
        self.device: torch.device | None = device
        self.arg_check: bool = arg_check
        self.sample_unsqueeze_default: bool = sample_unsqueeze
        self.theta_batch_first_default: bool = theta_batch_first
        self.sample_batch_first_default: bool = sample_batch_first

    # -----------------------------------------------------------------
    # validation helpers
    # -----------------------------------------------------------------
    def _validate_times(self, times: Tensor) -> None:
        """Validate a tensor of observation times.

        Checks performed
        ----------------
        * ``times`` is one-dimensional.
        * No element is ``NaN``.
        * Strictly increasing (no equal neighbours).
        * All entries are unique (duplicates raise an error).
        * If ``self.device`` is set, ``times`` must be on that device.

        Raises
        ------
        ValueError
            If any of the above conditions is violated.

        """
        if times.ndim != 1:
            raise ValueError("`times` must be a one-dimensional torch.Tensor")
        if torch.any(torch.isnan(times)):
            raise ValueError("`times` must not contain NaN")
        if not torch.all(times[1:] > times[:-1]):
            raise ValueError("`times` must be strictly increasing")
        if times.numel() != torch.unique(times).numel():
            raise ValueError("`times` must contain unique entries")
        if (self.device is not None) and times.device != self.device:
            raise ValueError(f"`times.device` is {times.device}, should be {self.device}")

    def _validate_theta(self, theta: Tensor) -> None:
        """Validate a tensor of ``theta`` arguments used for cumulants/characteristic
        functions.

        Checks performed
        ----------------
        * ``theta`` is two-dimensional.
        * No element is ``NaN``.
        * If ``self.device`` is set, ``theta`` must be on that device.

        Raises
        ------
        ValueError
            If either condition fails.

        """
        if theta.ndim != 2:
            raise ValueError("`theta` must be a two-dimensional torch.Tensor")
        if torch.any(torch.isnan(theta)):
            raise ValueError("`theta` must not contain NaN")
        if (self.device is not None) and theta.device != self.device:
            raise ValueError(f"`theta.device` is {theta.device}, should be {self.device}")

    def _validate_cumulant_args(
        self,
        theta: Tensor,
        times: Tensor,
        theta_batch_first: Optional[bool] = None,
    ) -> None:
        """Validate arguments passed to :meth:`cumulant` / :meth:`charfunc`.

        The validation is a superset of ``_validate_theta`` and
        ``_validate_times`` and additionally checks that the dimensions of
        ``theta`` and ``times`` are compatible with the supplied orientation.

        Parameters
        ----------
        theta : Tensor
            Fourier-argument tensor.
        times : Tensor
            Observation-time tensor.
        theta_batch_first : bool
            Orientation flag (``True`` -> ``(B, D)``, ``False`` -> ``(D, B)``).

        Raises
        ------
        ValueError
            If a shape or device incompatibility is detected.

        """
        self._validate_theta(theta)
        self._validate_times(times)

        if theta_batch_first is None:
            theta_batch_first = self.theta_batch_first_default

        if theta_batch_first:
            # (B, D) layout - ``D`` must match the number of observation times.
            if theta.shape[1] != times.shape[0]:
                raise ValueError("`theta.shape[1]` must equal `len(times)` when `theta_batch_first=True`")
        else:
            # (D, B) layout - ``D`` must match the number of observation times.
            if theta.shape[0] != times.shape[0]:
                raise ValueError("`theta.shape[0]` must equal `len(times)` when `theta_batch_first=False`")

    def _normalize_theta(self, theta: Tensor, theta_batch_first: Optional[bool] = None) -> Tensor:
        """Normalise ``theta`` to the internal ``(B, D)`` layout.

        Parameters
        ----------
        theta : Tensor
            Input tensor of shape either ``(B, D)`` (if ``theta_batch_first`` is
            ``True``) or ``(D, B)`` (if ``False``).
        theta_batch_first : bool, optional
            Overrides the default orientation stored on the process instance.
            If ``None`` the default value from ``self.theta_batch_first_default``
            is used.

        Returns
        -------
        Tensor
            ``theta`` reshaped to ``(B, D)``.

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
        """Normalise a sampled-trajectory tensor according to the library's conventions.

        The method performs two independent transformations that can be
        controlled on a per-call basis:

        1. **Batch-axis orientation** - if ``sample_batch_first`` is ``False``,
           the batch dimension is moved to the last axis.
        2. **Trailing unsqueeze** - if ``sample_unsqueeze`` is ``True``,
           a singleton dimension is appended at the end (e.g. ``(B, D)`` ->
           ``(B, D, 1)``).

        Parameters
        ----------
        data : Tensor
            Tensor returned by the underlying FDD's :meth:`sample`.  Its shape
            follows the orientation indicated by the FDD (normally ``(B, D, ...)``).
        sample_unsqueeze : bool, optional
            Overrides the default ``sample_unsqueeze`` flag stored on the
            process.  If ``True``, a trailing singleton dimension is added.
        sample_batch_first : bool, optional
            Overrides the default ``sample_batch_first`` flag stored on the
            process.  If ``False``, the batch axis is moved to the last
            dimension.

        Returns
        -------
        Tensor
            Normalised sample with shape ``(batch, time, 1)`` (or ``(time,
            batch, 1)`` when ``sample_batch_first=False``) unless the
            ``sample_unsqueeze`` flag is ``False``.

        """
        if sample_unsqueeze is None:
            sample_unsqueeze = self.sample_unsqueeze_default
        if sample_batch_first is None:
            sample_batch_first = self.sample_batch_first_default

        if not sample_batch_first:
            # ``data`` is expected to have shape (B, D, ...).  Swapping the first
            # two axes puts the batch dimension last.
            data = data.transpose(0, 1)

        if sample_unsqueeze:
            data = data.unsqueeze(-1)

        return data

    # -----------------------------------------------------------------
    # public API (delegates to a finite-dimensional-distribution object)
    # -----------------------------------------------------------------
    @abstractmethod
    def at_times(self, times: Tensor, rng: Optional[Generator] = None) -> StationaryProcessFDD:
        """Build a *finite-dimensional distribution* (FDD) object for the supplied
        observation grid ``times``.

        Sub-classes must return an instance of a concrete subclass of
        :class:`StationaryProcessFDD` that implements the heavy lifting for
        cumulants, characteristic functions and sampling.

        Parameters
        ----------
        times : Tensor
            A 1-D strictly increasing tensor of observation times.  The
            validation performed by :meth:`_validate_times` ensures monotonicity,
            finiteness and device compatibility.
        rng : torch.Generator, optional
            Optional random-number generator for reproducible sampling.

        Returns
        -------
        StationaryProcessFDD
            An object that knows the grid and the parent process.

        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self, x: Tensor) -> Tensor:
        """Marginal probability density function of the process.

        Sub-classes must provide a vector-valued (or scalar) density evaluated
        at the points ``x``.  The return shape should broadcast with ``x``.

        Parameters
        ----------
        x : Tensor
            Points at which to evaluate the density.

        Returns
        -------
        Tensor
            Density values with the same ``dtype`` as ``x`` and broadcastable
            shape.

        """
        raise NotImplementedError

    @abstractmethod
    def acf(self, lags: Tensor) -> Tensor:
        """Autocorrelation function (ACF) for the process.

        The implementation should return a tensor of autocorrelations for each
        lag in ``lags``.  The shape must broadcast with ``lags``.

        Parameters
        ----------
        lags : Tensor
            Non-negative lags (time differences) for which to compute the ACF.

        Returns
        -------
        Tensor
            Autocorrelation values, broadcastable to ``lags``.

        """
        raise NotImplementedError

    def cumulant(
        self,
        theta: Tensor,
        times: Tensor,
        theta_batch_first: Optional[bool] = None,
    ) -> Tensor:
        r"""Compute the joint cumulant.

        .. math::
            \\kappa(\\theta) = \\log \\mathbb E\bigl[\\exp(i\\langle \theta, X[\\mathrm{times}]\\rangle)\\bigr]

        for the supplied observation grid.

        The method validates the arguments, builds the appropriate FDD via
        :meth:`at_times`, and forwards the request to its :meth:`cumulant`
        implementation.

        Parameters
        ----------
        theta : Tensor
            Tensor of Fourier arguments.  Accepted layouts are ``(B, D)`` or
            ``(D, B)`` depending on ``theta_batch_first``.
        times : Tensor
            1-D tensor of observation times.
        theta_batch_first : bool, optional
            Overrides the class-level default orientation.  ``True`` indicates a
            ``(B, D)`` layout; ``False`` a ``(D, B)`` layout.

        Returns
        -------
        Tensor
            Tensor of shape ``(B,)`` containing the cumulant for each batch.

        """
        if theta_batch_first is None:
            theta_batch_first = self.theta_batch_first_default
        if self.arg_check:
            self._validate_cumulant_args(theta, times, theta_batch_first)
        return self.at_times(times).cumulant(theta, theta_batch_first)

    def charfunc(
        self,
        theta: Tensor,
        times: Tensor,
        theta_batch_first: Optional[bool] = None,
    ) -> Tensor:
        r"""Compute the joint characteristic function.

        .. math::
            \\vaprhi(\\theta) = \\exp\\bigl(\\kappa(\\theta)\\bigr),

        where :math: `\\kappa` is the cumulant computed by :meth:`cumulant`.

        Parameters
        ----------
        theta : Tensor
            Fourier-argument tensor (same layout conventions as :meth:`cumulant`).
        times : Tensor
            1-D observation-time tensor.
        theta_batch_first : bool, optional
            Orientation flag for ``theta``; defaults to the class setting.

        Returns
        -------
        Tensor
            Characteristic-function values, broadcastable to the shape of the
            input ``theta``.

        """
        return torch.exp(self.cumulant(theta, times, theta_batch_first))

    def sample(
        self,
        times: Tensor,
        *,
        rng: Optional[Generator] = None,
        batch_size: int = 1,
        batch_first: Optional[bool] = None,
        unsqueeze_last: Optional[bool] = None,
        **kwargs,
    ) -> Tensor:
        """Sample trajectories of the process on a user supplied grid.

        The method delegates the actual sampling to the :class:`StationaryProcessFDD`
        returned by :meth:`at_times`.  Any additional keyword arguments are
        forwarded verbatim to the FDD's ``sample`` method (e.g. a ``seed`` or
        implementation-specific options).

        Parameters
        ----------
        times : Tensor
            1-D increasing tensor of observation times.
        rng : torch.Generator, optional
            Optional generator to obtain deterministic draws.
        batch_size : int, default=1
            Number of independent trajectories to return.
        batch_first : bool, default=True
            If ``True`` (default) the output tensor has shape
            ``(batch_size, len(times), ...)``.  If ``False`` the batch dimension
            is placed last.
        unsqueeze_last : bool, default=True
            If ``True`` an additional singleton dimension is appended at the
            end (matching the original library convention).
        **kwargs
            Additional arguments passed directly to the underlying FDD's
            ``sample`` method.

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
