from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Concatenate, List, Optional, Sequence

import torch
from torch import LongTensor, Tensor
from tqdm.auto import tqdm

from ...processes import StationaryProcessFDD, StationaryStochasticProcess
from ..helpers import _normalize_data
from .base_loss import BaseStatLoss

# --------------------------------------------------------------------- #
# Module-level defaults (centralised constants)
# --------------------------------------------------------------------- #
DEFAULT_MC_POINTS: int = 1_000  # number of Monte-Carlo draws
DEFAULT_MC_BOUND: float = 1.0  # integration bound for uniform draw


def DEFAULT_KERNEL(theta: Tensor) -> Tensor:
    return torch.ones(theta.shape[0], device=theta.device)  # importance-sampling kernel


def DEFAULT_RISK(a: Tensor, b: Tensor) -> Tensor:
    return torch.abs(a - b)  # point-wise loss (|·|)


# --------------------------------------------------------------------- #
# A single component of the Char-Func loss
# --------------------------------------------------------------------- #
@dataclass
class CharFuncComponent:
    """A Monte-Carlo component of a characteristic-function loss.

    Attributes
    ----------
    idx : Tensor
        1-D integer tensor selecting the observation times that this component
        works on (e.g. ``torch.arange(i, i + window)``).
    target_fn : Callable[[Tensor], Tensor]
        The target characteristic function (either analytic from a model FDD or
        empirical via ``MonteCarloEstimatedCF``).  It must accept a tensor of
        shape ``(M, D)`` and return a complex tensor of shape ``(M,)``.

    """

    idx: Tensor
    target_fn: Callable[Concatenate[Tensor, ...], Tensor]

    def _mc_estimate(
        self,
        data: Tensor,
        data_batch_first: bool = False,
        *,
        data_kernel: Callable[[Tensor], Tensor] = DEFAULT_KERNEL,
        mc_points: int = DEFAULT_MC_POINTS,
        mc_bound: float = DEFAULT_MC_BOUND,
        risk: Callable[[Tensor, Tensor], Tensor] = DEFAULT_RISK,
    ) -> Tensor:
        """Monte-Carlo contribution of *this* component.

        Parameters
        ----------
        data_batched_first : Tensor
            Shape ``(B, D, 1)`` - batch-first representation of the full data.
        data_kernel : Callable, optional
            Importance-sampling kernel evaluated at the MC points.
        mc_points : int, optional
            Number of MC points (default ``DEFAULT_MC_POINTS``).
        mc_bound : float, optional
            Integration bound for the uniform draw (default ``DEFAULT_MC_BOUND``).
        risk : Callable, optional
            Point-wise loss between the empirical and target characteristic
            functions (default ``DEFAULT_RISK``).

        Returns
        -------
        Tensor
            Scalar (0-D) tensor that can be summed across components.

        """
        data = _normalize_data(data, data_batch_first)
        device = data.device

        def evaluate_loss(theta: Tensor) -> Tensor:
            "Evaluate loss"
            return risk(
                torch.mean(torch.exp(1.0j * torch.sum(data[:, self.idx, :] * theta.t(), dim=1)), dim=0)
                * data_kernel(theta),
                self.target_fn(theta, theta_batch_first=True),
            ).mean()

        # Random MC points uniformly drawn from [-bound, bound]^D
        return evaluate_loss(2.0 * mc_bound * torch.rand(mc_points, self.idx.shape[0], device=device) - mc_bound)

        # Evaluate both empirical and target CFs, compute the risk and average


# Helper - create a component from a model FDD or from empirical data
def _make_cf_component_from_fdd(
    idx: Tensor,
    fdd: StationaryProcessFDD,
) -> CharFuncComponent:
    """Build a :class:`CharFuncComponent` from a model's finite-dimensional
    distribution.

    Parameters
    ----------
    idx : Tensor
        1-D integer tensor of selected time indices.
    process : StationaryStochasticProcess
        The stochastic process model; ``process.at_times(times[idx])`` must
        return an object exposing a ``charfunc`` method.

    Returns
    -------
    CharFuncComponent

    """
    return CharFuncComponent(idx=idx, target_fn=fdd.charfunc)


def _make_cf_component_from_data(
    idx: Tensor,
    data: Tensor,
    *,
    kernel: Callable[[Tensor], Tensor] = DEFAULT_KERNEL,
) -> CharFuncComponent:
    """Build a :class:`CharFuncComponent` from observed data.

    Parameters
    ----------
    idx : Tensor
        1-D integer tensor of selected time indices.
    data : Tensor
        Observation matrix, shape ``(B, D, 1)`` (batch-first).
    kernel : Callable, optional
        Importance-sampling kernel for the empirical characteristic function.

    Returns
    -------
    CharFuncComponent

    """

    def target_fn(theta: Tensor, theta_batch_first: Optional[bool] = True) -> Tensor:
        if theta_batch_first:
            theta = theta.t()
        return torch.mean(torch.exp(1.0j * torch.sum(data[:, idx, :] * theta, dim=1)), dim=0) * kernel(theta.t())

    return CharFuncComponent(idx=idx, target_fn=target_fn)


class CharFuncLoss(BaseStatLoss):
    """Sum of losses given by :class:`CharFuncComponent`.

    The class knows how to turn user data into the required batch-first
    format, runs each component's MC estimator and (optionally) applies
    a Jacobian normalisation factor.
    """

    def __init__(
        self,
        components: Sequence[CharFuncComponent],
        *,
        risk: Callable[[Tensor, Tensor], Tensor] = DEFAULT_RISK,
        data_kernel: Callable[[Tensor], Tensor] = DEFAULT_KERNEL,
        mc_points: int = DEFAULT_MC_POINTS,
        mc_bound: float = DEFAULT_MC_BOUND,
        dim_normalization: bool = True,
        data_batch_first: bool = False,
        disable_tqdm: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(torch.tensor(0.0), lambda t: torch.tensor(0.0), **kwargs)  # type: ignore
        if not components:
            raise ValueError("CharFuncLoss needs at least one component")
        self.components = list(components)

        self.data_batch_first_default = data_batch_first
        self.disable_tqdm_default = disable_tqdm

        # Store defaults for the MC estimator
        self.risk = risk
        self.data_kernel = data_kernel
        self.mc_points = mc_points
        self.mc_bound = mc_bound
        self.dim_normalization = dim_normalization

    # -----------------------------------------------------------------
    # 2. Forward - sum over components
    # -----------------------------------------------------------------
    def forward(
        self,
        data: Tensor,
        *,
        data_batch_first: Optional[bool] = None,
        disable_tqdm: Optional[bool] = None,
    ) -> Tensor:
        """Evaluate the loss.

        Parameters
        ----------
        data : Tensor
            Either ``(B, D)`` / ``(B, D, 1)`` (batch-first) or the transposed
            version.  The function automatically reshapes it.
        data_batch_first : bool, optional
            Overrides the default supplied at construction.  If ``None`` the
            default is used.
        disable_tqdm : bool, default True
            Whether to hide the progress bar.

        Returns
        -------
        Tensor
            Scalar loss (0-D tensor).

        """
        if data_batch_first is None:
            data_batch_first = self.data_batch_first_default
        if disable_tqdm is None:
            disable_tqdm = self.disable_tqdm_default

        data = _normalize_data(data, data_batch_first)
        loss = torch.tensor(0.0, device=data.device, dtype=data.dtype)
        for comp in tqdm(self.components, disable=disable_tqdm):
            jacob = 1.0
            if self.dim_normalization:
                # Jacobian factor (2·bound)^{|idx|}
                jacob = (2.0 * self.mc_bound) ** comp.idx.shape[0]
            loss += jacob * comp._mc_estimate(
                data,
                data_batch_first=True,
                data_kernel=self.data_kernel,
                mc_points=self.mc_points,
                mc_bound=self.mc_bound,
                risk=self.risk,
            )
        return loss

    @classmethod
    def _build_comp_idxs(cls, D: int, **loss_kwargs) -> List[LongTensor]:
        raise NotImplementedError

    @classmethod
    def analytical(cls, times: Tensor, process: StationaryStochasticProcess, **loss_kwargs) -> CharFuncLoss:
        idxs = cls._build_comp_idxs(times.shape[0], **loss_kwargs)
        components = []
        for idx in idxs:
            components.append(_make_cf_component_from_fdd(idx, process.at_times(times[idx])))
        return cls(components, **loss_kwargs)

    @classmethod
    def empirical(cls, data: Tensor, **loss_kwargs) -> CharFuncLoss:
        kernel = loss_kwargs.get("data_kernel", DEFAULT_KERNEL)
        data_batch_first = loss_kwargs.get("data_batch_first", False)

        data_ = _normalize_data(data, data_batch_first)
        idxs = cls._build_comp_idxs(data_.shape[1])
        components = []
        for idx in idxs:
            components.append(_make_cf_component_from_data(idx, data_, kernel=kernel))
        return cls(components, **loss_kwargs)
