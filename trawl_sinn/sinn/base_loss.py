#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Statistical loss functions for stochastic‑process modelling.

All concrete losses inherit from :class:`BaseStatLoss`.  Several losses can be
combined with standard arithmetic operators:

>>> loss = 2 * acf_loss + 0.5 * density_loss   # → WeightedStatLoss
>>> total = loss(model(x))                    # scalar

Typical usage
~~~~~~~~~~~~~
>>> # Empirical trajectories (T, B, D)
>>> data = torch.randn(200, 64, 3)
>>> loss = make_loss(
...     stat="acf[fft]",
...     data=data,
...     loss_type="mse_loss",          # ← only a *single* loss is allowed
...     lags=30,
... )
>>> # ``model`` returns a tensor of shape (T, B, D)
>>> loss(model(x))   # → scalar loss
"""

from __future__ import annotations

from typing import Callable, Sequence, Union, Concatenate, cast, Any, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Loss resolution
# --------------------------------------------------------------------------- #

LossFn = Callable[Concatenate[Tensor, Tensor, ...], Tensor]


def _resolve_loss(loss: Union[LossFn, str]) -> LossFn:
    """Translate a string such as ``"mse_loss"`` to the corresponding functional.

    Parameters
    ----------
    loss :
        Either a callable ``loss(pred, target, reduction=…)`` or the name of a
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
    """
    Generic wrapper that

    1. Computes a statistic on the network output (``stat_fn``);
    2. Compares it to a pre‑computed target using a **single** point‑wise loss
       function.

    Sub‑classes only need to supply ``stat_fn`` (e.g. ``acf_fft``) and a target
    tensor.  All boiler‑plate (device handling, loss aggregation, operator
    overloads) lives here.

    Parameters
    ----------
    target :
        The reference statistic.  It is registered as a buffer so that it
        automatically moves with ``.to(device)``.
    stat_fn :
        Callable ``stat_fn(x)`` that returns the statistic for a network output
        ``x``.  The returned tensor must have the same shape as ``target``.
    pointwise_loss :
        Either a callable loss ``loss(pred, target, reduction=…)`` **or** the
        name of a function in :mod:`torch.nn.functional` (e.g. ``"mse_loss"``).
        Only a *single* loss is accepted – if you need to combine several,
        use :class:`WeightedStatLoss` explicitly.
    reduction :
        Passed through to the underlying functional loss (``"mean"``,
        ``"sum"`` or ``"none"``).
    **extra_attrs :
        Arbitrary keyword arguments are stored as attributes on the instance.
        This is convenient for keeping hyper‑parameters (e.g. KDE bandwidth).
    """

    def __init__(
        self,
        target: Tensor,
        stat_fn: Callable[[Tensor], Tensor],
        *,
        loss_fn: Union[LossFn, str] = "mse_loss",
        loss_fn_opts: Dict[str, Any] = {},
        data_batch_first: bool = False,
        **extra_attrs,
    ) -> None:
        super().__init__()
        # Register target as a buffer so it follows the module’s device.
        self.register_buffer("_target", target)
        self._target: Tensor = cast(Tensor, self._target)  # MyPy trick

        self.stat_fn = stat_fn
        self.loss_fn: LossFn = _resolve_loss(loss_fn)
        self.loss_fn_opts: Dict[str, Any] = loss_fn_opts

        self.data_batch_first = data_batch_first 

        # Store auxiliary attributes (e.g. KDE bandwidth, random‑lag mask, …).
        for name, value in extra_attrs.items():
            setattr(self, name, value)

    # --------------------------------------------------------------------- #
    #  Forward pass
    # --------------------------------------------------------------------- #

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the statistic on ``x`` and apply the configured point‑wise loss.

        Parameters
        ----------
        x :
            Tensor with the same shape as the model output on which the
            statistic is evaluated.

        Returns
        -------
        Tensor
            A scalar (or a zero‑dimensional tensor) that can be back‑propagated.
        """
        x = x.squeeze()
        if x.ndim == 3:
            raise ValueError("Multidimensional processes are not supported yet")
        if self.data_batch_first:
            x = x.swapaxes(0, 1)
        return self.loss_fn(self.stat_fn(x), self._target, **self.loss_fn_opts)

    # --------------------------------------------------------------------- #
    #  Arithmetic operator overloads – enable easy loss composition
    # --------------------------------------------------------------------- #

    def __add__(
        self, other: Union["BaseStatLoss", "WeightedStatLoss"]
    ) -> "WeightedStatLoss":
        """
        ``self + other`` → a :class:`WeightedStatLoss` that adds the two
        components with unit weight.

        The right‑hand side may be another :class:`BaseStatLoss`,
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

    def __rmul__(self, weight: float) -> "WeightedStatLoss":
        """``weight * loss`` – create a :class:`WeightedStatLoss` with a single
        term whose weight equals *weight*."""
        if not isinstance(weight, (int, float)):
            return NotImplemented
        w = torch.tensor(
            [float(weight)],
            dtype=self._target.dtype,
            device=self._target.device,
        )
        return WeightedStatLoss([self], w)

    # Enable ``loss * weight`` (the left‑hand side) as a friendly shortcut.
    __mul__ = __rmul__


# --------------------------------------------------------------------------- #
#  Weighted loss container
# --------------------------------------------------------------------------- #


class WeightedStatLoss(nn.Module):
    """
    Container that aggregates a list of loss modules, each optionally scaled by
    a scalar weight.  The forward pass evaluates::

        Σ_i weight_i × loss_i(x)

    where each ``loss_i`` is an instance of :class:`BaseStatLoss` (or any
    ``nn.Module`` that returns a scalar tensor).

    Parameters
    ----------
    terms :
        Sequence of ``BaseStatLoss`` objects that compute a scalar loss.
    weights :
        Sequence (or 1‑D ``torch.Tensor``) of the same length as *terms*
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
            raise ValueError("`weights` must be a 1‑dimensional tensor or sequence.")
        if len(w_tensor) != len(terms):
            raise ValueError(
                f"Number of weights ({len(w_tensor)}) does not match number of "
                f"terms ({len(terms)})."
            )
        # Buffer registration guarantees correct device handling.
        self.register_buffer("weights", w_tensor)
        self.weights: Tensor = cast(Tensor, self.weights)  # MyPy trick

        self.terms = terms

    # --------------------------------------------------------------------- #
    #  Arithmetic operator overloads – keep the API symmetric with BaseStatLoss
    # --------------------------------------------------------------------- #

    def __add__(
        self, other: Union[BaseStatLoss, "WeightedStatLoss"]
    ) -> "WeightedStatLoss":
        """``self + other`` – return a new :class:`WeightedStatLoss` that
        concatenates the two lists of terms and appends a unit weight for any
        plain :class:`BaseStatLoss`."""
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

    def __radd__(
        self, other: Union[BaseStatLoss, "WeightedStatLoss"]
    ) -> "WeightedStatLoss":
        return self.__add__(other)

    def __mul__(self, weight: float) -> "WeightedStatLoss":
        """``self * weight`` – return a new container with all weights scaled."""
        if not isinstance(weight, (int, float)):
            return NotImplemented
        new_weights = self.weights * float(weight)
        return WeightedStatLoss(list(self.terms), new_weights)

    __rmul__ = __mul__

    # --------------------------------------------------------------------- #
    #  Forward evaluation
    # --------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the weighted sum of the individual loss modules.

        Parameters
        ----------
        x :
            Input tensor that will be fed to each sub‑loss module.

        Returns
        -------
        Tensor
            Scalar loss (zero‑dimensional tensor) equal to
            ``Σ_i weight_i × loss_i(x)``.
        """
        total = torch.tensor(
            [loss_mod(x) for loss_mod in self.terms],
            dtype=x.dtype,
            device=x.device,
        )
        return torch.sum(self.weights * total)


# --------------------------------------------------------------------------- #
#  Exported names
# --------------------------------------------------------------------------- #

__all__ = [
    "BaseStatLoss",
    "WeightedStatLoss",
    "LossFn",
]
