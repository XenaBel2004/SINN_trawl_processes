#!/usr/bin/env python
# -*- coding: ascii -*-
"""Statistical loss functions for stochastic-process modelling.

All concrete losses inherit from :class:`BaseStatLoss`.  Several losses can be
combined with standard arithmetic operators:

>>> loss = 2 * acf_loss + 0.5 * density_loss   # -> WeightedStatLoss
>>> total = loss(model(x))                     # scalar

Typical usage
~~~~~~~~~~~~~
>>> # Empirical trajectories (T, B, D)
>>> data = torch.randn(200, 64, 3)
>>> loss = make_loss(
...     stat="acf[fft]",
...     data=data,
...     loss_type="mse_loss",
...     lags=30,
... )
>>> # ``model`` returns a tensor of shape (T, B, D)
>>> loss(model(x))   # -> scalar loss
"""

from __future__ import annotations

from typing import Sequence, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...utils import BatchedTensorFn, LossFn

# --------------------------------------------------------------------------- #
#  Loss resolution
# --------------------------------------------------------------------------- #


def _resolve_loss(loss: Union[LossFn, str]) -> LossFn:
    """Translate a string such as ``"mse_loss"`` to the corresponding
    functional.

    Parameters
    ----------
    loss :
        Either a callable ``loss(pred, target, reduction=...)`` or the name of a
        function in :mod:`torch.nn.functional` (e.g. ``"mse_loss"``).

    Returns
    -------
    Callable
        The resolved loss function.

    Raises
    ------
    TypeError
        If *loss* is neither callable nor a string.
    ValueError
        If the string does not correspond to a function in ``torch.nn.functional``.

    """
    if callable(loss):
        return loss
    if not isinstance(loss, str):
        raise TypeError(f"Loss must be a callable or a string, got {type(loss)}")
    try:
        loss_func = getattr(F, loss)

        def loss(x: Tensor, y: Tensor):
            return loss_func(x, y, reduction="mean")

        return loss

    except AttributeError:  # pragma: no cover
        raise ValueError(f"torch.nn.functional has no loss named '{loss}'")


# --------------------------------------------------------------------------- #
#  Base loss wrapper
# --------------------------------------------------------------------------- #


class BaseStatLoss(nn.Module):
    """Generic wrapper that computes a statistic on the model output and
    compares it to a target using a single point-wise loss function.

    The wrapper handles device registration for the ``target`` statistic,
    optional transposition of the input when ``data_batch_first`` is ``True``,
    and provides arithmetic operator overloads for easy composition of multiple
    statistics. Sub-classes only need to implement ``stat_fn`` (the statistic
    computation) and supply a pre-computed ``target`` tensor; all other
    boiler-plate is managed here.

    Parameters
    ----------
    target : torch.Tensor
        Reference statistic.  It is registered as a buffer so that it automatically moves with
        ``.to(device)``. Its shape must exactly match the output of ``stat_fn``.
    stat_fn : BatchedTensorFn
        Callable ``stat_fn(x)`` that computes the desired statistic from a tensor ``x``.
        After any necessary transposition (see ``data_batch_first``), ``x`` is expected to have shape
        ``(time, batch, ...)``; the returned tensor must have the same shape as ``target``.
    loss_fn : Union[LossFn, str], optional
        Either a callable loss ``loss(pred, target, ...)`` or the name of a function in
        :mod:`torch.nn.functional` (e.g. ``"mse_loss"``).  If a string is given, it is
        resolved to the corresponding functional and called with ``reduction="mean"`` by default.
    data_batch_first : bool, default ``False``
        If ``True``, the input tensor to :meth:`forward` is expected to have shape
        ``(batch, time, ...)`` and will be swapped to ``(time, batch, ...)`` before ``stat_fn``
        is invoked.
    **extra_attrs**
        Arbitrary keyword arguments are stored as attributes on the loss instance.
        This is convenient for attaching hyper-parameters (e.g. ``lags=30`` or ``bandwidth=0.2``)
        for bookkeeping.

    """

    def __init__(
        self,
        target: Tensor,
        stat_fn: BatchedTensorFn,
        *,
        loss_fn: Union[LossFn, str] = "mse_loss",
        data_batch_first: bool = False,
        **extra_attrs,
    ) -> None:
        """Initializes BaseStatLoss object.

        Parameters
        ----------
        target : torch.Tensor
            Reference statistic against which the learned statistic will be compared.
            The tensor is registered as a buffer, so it automatically moves with the
            module when ``.to(device)`` is called. Its shape must exactly match the
            output of ``stat_fn`` (typically a 1-D tensor for univariate statistics).

        stat_fn : BatchedTensorFn
            Callable ``stat_fn(x)`` that computes the desired statistic from a tensor
            ``x``.  Internaly supplied ``x`` in ``meth:forward`` will always have
            shape ``(batch, time ...)``; the returned tensor must have the same shape
            as ``target``.

        loss_fn : Union[LossFn, str], optional
            Either a callable loss ``loss(pred, target, ...)`` or the name of a function
            in :mod:`torch.nn.functional` (e.g. ``"mse_loss"``).  If a string is given,
            it is resolved to the corresponding functional and called with
            ``reduction="mean"`` by default.

        data_batch_first : bool, default ``False``
            If ``True``, the input tensor to :meth:`forward` is expected to have shape
            ``(batch, time, ...)`` and will be swapped to ``(time, batch, ...)``
            before ``stat_fn`` is invoked.

        **extra_attrs**
            Arbitrary keyword arguments are stored as attributes on the loss instance.
            This is convenient for attaching hyper-parameters (e.g. ``lags=30`` or
            ``bandwidth=0.2``) that may be required by ``stat_fn`` or for reference.

        Returns
        -------
        None

        """
        super().__init__()
        self.register_buffer("_target", target)
        self._target: Tensor = cast(Tensor, self._target)  # MyPy trick

        self.stat_fn: BatchedTensorFn = stat_fn
        self.loss_fn: LossFn = _resolve_loss(loss_fn)
        self.data_batch_first = data_batch_first

        # Store auxiliary attributes (e.g. KDE bandwidth, ACF lags ...).
        for name, value in extra_attrs.items():
            setattr(self, name, value)

    # --------------------------------------------------------------------- #
    #  Forward pass
    # --------------------------------------------------------------------- #

    def forward(self, x: Tensor) -> Tensor:
        """Compute the statistic on ``x`` and evaluate the configured loss.

        Parameters
        ----------
        x : torch.Tensor
            Model output to be evaluated.  Accepted shapes (before squeezing) are
            ``(batch, time)``, ``(batch, time, 1)``, ``(time, batch)`` or
            ``(time, batch, 1)``.

        Returns
        -------
        torch.Tensor
            A scalar (zero-dimensional) tensor representing the loss between
            the computed statistic and the target.

        Raises
        ------
        ValueError
            if ``x`` have incorrect shape

        """
        x = x.squeeze(dim=2)
        if x.ndim >= 3:
            raise ValueError("Multidimensional processes are not supported yet")

        if not self.data_batch_first:
            x = x.swapaxes(0, 1)
        return self.loss_fn(self.stat_fn(x), self._target)

    def __add__(self, other: Union[BaseStatLoss, WeightedStatLoss]) -> WeightedStatLoss:
        """``self + other`` -> a :class:`WeightedStatLoss` that adds the two
        components with unit weight.

        The right-hand side may be another :class:`BaseStatLoss`,
        a :class:`WeightedStatLoss` or ``0`` (used by :func:`sum`).  ``0 +
        self`` is handled by :meth:`__radd__`.
        """
        if isinstance(other, BaseStatLoss):
            terms = [self, other]
            weights = torch.tensor(
                [1.0, 1.0],
                dtype=self._target.dtype,
                device=self._target.device,
            )
            return WeightedStatLoss(terms, weights)

        if isinstance(other, WeightedStatLoss):
            terms = [self] + list(other.terms)  # type: ignore
            weights = torch.cat(
                [
                    torch.tensor(
                        [1.0],
                        dtype=self._target.dtype,
                        device=self._target.device,
                    ),
                    other.weights,
                ]
            )
            return WeightedStatLoss(terms, weights)

        return NotImplemented

    def __rmul__(self, weight: float) -> WeightedStatLoss:
        """``weight * loss`` - create a :class:`WeightedStatLoss` with a single
        term whose weight equals *weight*.
        """
        if not isinstance(weight, (int, float)):
            return NotImplemented
        w = torch.tensor(
            [float(weight)],
            dtype=self._target.dtype,
            device=self._target.device,
        )
        return WeightedStatLoss([self], w)

    __mul__ = __rmul__


# --------------------------------------------------------------------------- #
#  Weighted loss container
# --------------------------------------------------------------------------- #


class WeightedStatLoss(nn.Module):
    """Container that aggregates a list of loss modules, each optionally scaled by
    a scalar weight.  The forward pass evaluates::

        Sum[i = 1 ... ] weight_i * loss_i(x)

    where each ``loss_i`` is an instance of :class:`BaseStatLoss` (or any
    ``nn.Module`` that returns a scalar tensor).

    Parameters
    ----------
    terms :
        Sequence of ``BaseStatLoss`` objects that compute a scalar loss.
    weights :
        Sequence (or 1-D ``torch.Tensor``) of the same length as *terms*
        containing the scalar weights.  The tensor is stored as a buffer so it
        follows the module's device.

    """

    def __init__(
        self,
        terms: Sequence[BaseStatLoss],
        weights: Union[Sequence[float], torch.Tensor],
    ) -> None:
        super().__init__()
        if len(terms) == 0:
            raise ValueError("WeightedStatLoss requires at least one term.")
        if isinstance(weights, torch.Tensor):
            w_tensor = weights.detach().clone()
        else:
            w_tensor = torch.tensor(weights, dtype=torch.float32)

        if w_tensor.ndim != 1:
            raise ValueError("`weights` must be a 1-dimensional tensor or sequence.")
        if len(w_tensor) != len(terms):
            raise ValueError(f"Number of weights ({len(w_tensor)}) does not match number of terms ({len(terms)}).")
        # Buffer registration guarantees correct device handling.
        self.register_buffer("weights", w_tensor)
        self.weights: Tensor = cast(Tensor, self.weights)  # MyPy trick

        self.terms = terms

    def __add__(self, other: Union[BaseStatLoss, WeightedStatLoss]) -> "WeightedStatLoss":
        """``self + other`` - return a new :class:`WeightedStatLoss` that
        concatenates the two lists of terms and appends a unit weight for any
        plain :class:`BaseStatLoss`.
        """
        if isinstance(other, BaseStatLoss):
            new_terms = list(self.terms) + [other]
            new_weights = torch.cat(
                [
                    self.weights,
                    torch.tensor(
                        [1.0],
                        dtype=self.weights.dtype,
                        device=self.weights.device,
                    ),
                ]
            )
            return WeightedStatLoss(new_terms, new_weights)

        if isinstance(other, WeightedStatLoss):
            new_terms = list(self.terms) + list(other.terms)
            new_weights = torch.cat([self.weights, other.weights])
            return WeightedStatLoss(new_terms, new_weights)

        return NotImplemented

    def __radd__(self, other: Union[BaseStatLoss, WeightedStatLoss]) -> WeightedStatLoss:
        return self.__add__(other)

    def __mul__(self, weight: float) -> WeightedStatLoss:
        """``self * weight`` - return a new container with all weights scaled."""
        if not isinstance(weight, (int, float)):
            return NotImplemented
        new_weights = self.weights * float(weight)
        return WeightedStatLoss(list(self.terms), new_weights)

    __rmul__ = __mul__

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the weighted sum of the individual loss modules.

        Parameters
        ----------
        x :
            Input tensor that will be fed to each sub-loss module.

        Returns
        -------
        Tensor
            Scalar loss (zero-dimensional tensor) equal to
            ``Sum[i = 1 ...] weight_i * loss_i(x)``.

        """
        total = torch.tensor(
            [loss_mod(x) for loss_mod in self.terms],
            dtype=x.dtype,
            device=x.device,
        )
        return torch.sum(self.weights * total)
