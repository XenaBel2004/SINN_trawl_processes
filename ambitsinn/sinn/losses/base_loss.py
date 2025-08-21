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

from typing import Union, cast

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...utils import BatchedTensorFn, LossFn
from ..helpers import _normalize_data

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

    def forward(self, data: Tensor) -> Tensor:
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
        data = _normalize_data(data, self.data_batch_first)
        return self.loss_fn(self.stat_fn(data), self._target)
