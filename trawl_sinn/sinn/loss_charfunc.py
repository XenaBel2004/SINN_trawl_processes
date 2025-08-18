# ──────────────────────────────────────────────────────────────────────
# charfunc_losses.py
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, List, Optional, Concatenate

import torch
from torch import Tensor
from .base_loss import BaseStatLoss
from ..processes import StationaryProcessFDD, StationaryStochasticProcess
from tqdm.auto import tqdm

# --------------------------------------------------------------------- #
# Module‑level defaults (centralised constants)
# --------------------------------------------------------------------- #
DEFAULT_MC_POINTS: int = 1_000  # number of Monte‑Carlo draws
DEFAULT_MC_BOUND: float = 1.0  # integration bound for uniform draw
def DEFAULT_KERNEL(theta: Tensor) -> Tensor:
    return torch.ones(
    theta.shape[0], device=theta.device
)  # importance‑sampling kernel
def DEFAULT_RISK(a: Tensor, b: Tensor) -> Tensor:
    return torch.abs(
    a - b
)  # point‑wise loss (|·|)


# --------------------------------------------------------------------- #
# Helper – MC estimator of a (possibly weighted) characteristic function
# --------------------------------------------------------------------- #


def _normalize_data(data: Tensor, is_batch_first: bool = False):
    """
    Convert ``data`` to shape ``(B, D, 1)`` (batch‑first).

    Accepted input shapes:
    * ``(B, D)``            – interpreted as batch‑first, a trailing 1 is added
    * ``(B, D, 1)``        – already OK
    * ``(D, B)``            – transposed, then a trailing 1 is added
    * ``(D, B, 1)``        – transposed
    """
    if data.ndim == 2:
        # (B, D)  or (D, B)
        if is_batch_first:
            out = data.unsqueeze(-1)  # (B, D, 1)
        else:
            out = data.t().unsqueeze(-1)  # (B, D, 1)
    elif data.ndim == 3 and data.shape[2] == 1:
        # (B, D, 1)  or (D, B, 1)
        if is_batch_first:
            out = data
        else:
            out = data.transpose(0, 1)  # swap batch & time
    else:
        raise ValueError(
            f"data must be 2‑D (B×D) or 3‑D (B×D×1); got shape {tuple(data.shape)}"
        )
    return out.contiguous()


class MonteCarloEstimatedCF:
    """
    Empirical characteristic function estimator (optionally weighted).

    Parameters
    ----------
    data_batched_first : Tensor
        Shape ``(B, D)`` where ``B`` is the batch size and ``D`` the number of
        selected observation times.
    kernel : Callable[[Tensor], Tensor], optional
        Importance‑sampling kernel ``k(θ)``.  By default it returns a vector of
        ones (i.e. no weighting). θ must be tensor of shape (M, D) where M is
        batch dimension.
    """

    def __init__(
        self,
        data: Tensor,
        *,
        data_batch_first: bool = False,
        kernel: Callable[[Tensor], Tensor] = DEFAULT_KERNEL,
    ) -> None:
        self.data = _normalize_data(data, data_batch_first)  # (B, D, 1)
        self.kernel = kernel

    def __call__(self, theta: Tensor, theta_batch_first: bool = True) -> Tensor:
        """
        Return the (kernel‑weighted) empirical characteristic function at ``θ``.

        Parameters
        ----------
        theta : Tensor
            Shape ``(D, M)`` where ``D`` is the dimensionality of the idx
            and ``M`` the number of Monte‑Carlo points.

        Returns
        -------
        Tensor
            Shape ``(M,)`` – the empirical CF evaluated at each MC point.
        """
        if not theta_batch_first:
            theta = theta.t()
        kern = self.kernel(theta)  # (M,)
        cumulant = 1.0j * (self.data[:, :, 0] @ theta.t())  # (B, M)
        charfunc = torch.mean(torch.exp(cumulant), dim=0)  # (M,)  complex
        return charfunc * kern


# --------------------------------------------------------------------- #
# A single component of the Char‑Func loss
# --------------------------------------------------------------------- #
@dataclass
class CharFuncComponent:
    """
    A Monte‑Carlo component of a characteristic‑function loss.

    Attributes
    ----------
    idx : Tensor
        1‑D integer tensor selecting the observation times that this component
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
        """
        Monte‑Carlo contribution of *this* component.

        Parameters
        ----------
        data_batched_first : Tensor
            Shape ``(B, D, 1)`` – batch‑first representation of the full data.
        data_kernel : Callable, optional
            Importance‑sampling kernel evaluated at the MC points.
        mc_points : int, optional
            Number of MC points (default ``DEFAULT_MC_POINTS``).
        mc_bound : float, optional
            Integration bound for the uniform draw (default ``DEFAULT_MC_BOUND``).
        risk : Callable, optional
            Point‑wise loss between the empirical and target characteristic
            functions (default ``DEFAULT_RISK``).

        Returns
        -------
        Tensor
            Scalar (0‑D) tensor that can be summed across components.
        """
        data_batched_first = _normalize_data(data, data_batch_first)
        device = data_batched_first.device
        _, D, _ = data_batched_first.shape

        # Random MC points uniformly drawn from [-bound, bound]^D
        theta = (
            2.0 * mc_bound * torch.rand(mc_points, self.idx.shape[0], device=device)
            - mc_bound
        )

        # Slice the data for the selected indices
        data_slice = data_batched_first[:, self.idx, :]  # (B, L, 1)
        empirical_cf = MonteCarloEstimatedCF(
            data_slice, data_batch_first=True, kernel=data_kernel
        )

        # Evaluate both empirical and target CFs, compute the risk and average
        loss = risk(
            empirical_cf(theta, theta_batch_first=True),
            self.target_fn(theta, theta_batch_first=True),
        ).mean()
        return loss


# Helper – create a component from a model FDD or from empirical data
def _make_cf_component_from_fdd(
    idx: Tensor,
    fdd: StationaryProcessFDD,
) -> CharFuncComponent:
    """
    Build a :class:`CharFuncComponent` from a model's finite‑dimensional
    distribution.

    Parameters
    ----------
    idx : Tensor
        1‑D integer tensor of selected time indices.
    process : StationaryStochasticProcess
        The stochastic process model; ``process.at_times(times[idx])`` must
        return an object exposing a ``charfunc`` method.

    Returns
    -------
    CharFuncComponent
    """

    return CharFuncComponent(idx=idx.to(fdd.process.device), target_fn=fdd.charfunc)


def _make_cf_component_from_data(
    idx: Tensor,
    data: Tensor,
    *,
    kernel: Callable[[Tensor], Tensor] = DEFAULT_KERNEL,
) -> CharFuncComponent:
    """
    Build a :class:`CharFuncComponent` from observed data.

    Parameters
    ----------
    idx : Tensor
        1‑D integer tensor of selected time indices.
    data : Tensor
        Observation matrix, shape ``(B, D, 1)`` (batch‑first).
    kernel : Callable, optional
        Importance‑sampling kernel for the empirical characteristic function.

    Returns
    -------
    CharFuncComponent
    """
    target_fn = MonteCarloEstimatedCF(
        data[:, idx, :], data_batch_first=True, kernel=kernel
    )
    return CharFuncComponent(idx=idx.to(data.device), target_fn=target_fn)


# --------------------------------------------------------------------- #
# Composite loss – weighted sum of components
# --------------------------------------------------------------------- #
class CharFuncLoss(BaseStatLoss):
    """
    Sum of losses given by :class:`CharFuncComponent`

    The class knows how to turn user data into the required batch‑first format,
    runs each component's MC estimator and (optionally) applies a Jacobian
    normalisation factor.
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
        self.dim_normalization = dim_normalization
        self.disable_tqdm_default = disable_tqdm

        # Store defaults for the MC estimator
        self.risk = risk
        self.data_kernel = data_kernel
        self.mc_points = mc_points
        self.mc_bound = mc_bound

    # -----------------------------------------------------------------
    # 2. Forward – sum over components
    # -----------------------------------------------------------------
    def forward(
        self,
        data: Tensor,
        *,
        data_batch_first: Optional[bool] = None,
        disable_tqdm: Optional[bool] = None,
    ) -> Tensor:
        """
        Evaluate the loss.

        Parameters
        ----------
        data : Tensor
            Either ``(B, D)`` / ``(B, D, 1)`` (batch‑first) or the transposed
            version.  The function automatically reshapes it.
        data_batch_first : bool, optional
            Overrides the default supplied at construction.  If ``None`` the
            default is used.
        disable_tqdm : bool, default True
            Whether to hide the progress bar.

        Returns
        -------
        Tensor
            Scalar loss (0‑D tensor).
        """
        if data_batch_first is None:
            data_batch_first = self.data_batch_first_default
        if disable_tqdm is None:
            disable_tqdm = self.disable_tqdm_default

        data_ = _normalize_data(data, data_batch_first)
        loss = torch.tensor(0.0, device=data_.device, dtype=data_.dtype)
        for comp in tqdm(self.components, disable=disable_tqdm):
            term = comp._mc_estimate(
                data_,
                data_batch_first=True,
                data_kernel=self.data_kernel,
                mc_points=self.mc_points,
                mc_bound=self.mc_bound,
                risk=self.risk,
            )
            if self.dim_normalization:
                # Jacobian factor (2·bound)^{|idx|}
                term = ((2.0 * self.mc_bound) ** comp.idx.shape[0]) * term
            loss = loss + term
        return loss

    @classmethod
    def _build_comp_idxs(cls, D: int, **loss_kwargs) -> List[Tensor]:
        raise NotImplementedError

    @classmethod
    def analytical(
        cls, times: Tensor, process: StationaryStochasticProcess, **loss_kwargs
    ) -> "CharFuncLoss":
        idxs = cls._build_comp_idxs(times.shape[0], **loss_kwargs)
        components = []
        for idx in idxs:
            components.append(
                _make_cf_component_from_fdd(idx, process.at_times(times[idx]))
            )
        return cls(components, **loss_kwargs)

    @classmethod
    def empirical(cls, data: Tensor, **loss_kwargs) -> "CharFuncLoss":
        kernel = loss_kwargs.get("data_kernel")
        if kernel is None:
            kernel = DEFAULT_KERNEL

        data_batch_first = loss_kwargs.get("data_batch_first")
        if data_batch_first is None:
            data_batch_first = False

        data_ = _normalize_data(data, data_batch_first)
        idxs = cls._build_comp_idxs(data_.shape[1])
        components = []
        for idx in idxs:
            components.append(_make_cf_component_from_data(idx, data_, kernel=kernel))
        return cls(components, **loss_kwargs)


# --------------------------------------------------------------------- #
# Concrete loss subclasses – each implements a specific “scheme” and offers
# factory classmethods ``from_fdd`` and ``from_empirical``.
# --------------------------------------------------------------------- #
class CFFullLoss(CharFuncLoss):
    """Single component that uses *all* observation times."""

    @classmethod
    def _build_comp_idxs(cls, D: int, **kwargs) -> List[Tensor]:
        return torch.arange(D)  # type: ignore


class CFPairwiseLoss(CharFuncLoss):
    """All unordered pairs of observation times (i < j)."""

    @classmethod
    def _build_comp_idxs(cls, D: int, **kwargs) -> List[Tensor]:
        return [torch.tensor([i, j]) for i in range(D - 1) for j in range(i + 1, D)]


class CFMarginalLoss(CharFuncLoss):
    """One component per single observation time (marginals)."""

    @classmethod
    def _build_comp_idxs(cls, D: int, **kwargs) -> List[Tensor]:
        return [torch.tensor([i]) for i in range(D)]  # type: ignore


class CFLongTermPairsLoss(CharFuncLoss):
    """
    Sparse “long‑term’’ pairs scheme: from each start index ``i`` (stride ``step``)
    pair it with ``j = i + d, i + 2·d, …``.
    """

    @classmethod
    def _build_comp_idxs(
        cls, D: int, roll_step: int = 10, pair_step: int = 10, **kwargs
    ) -> List[Tensor]:
        pairs: List[Tensor] = []
        for i in range(0, D - 1, roll_step):
            for j in range(i + pair_step, D, pair_step):
                pairs.append(torch.tensor([i, j]))  # type: ignore
        return pairs


class CFRollingWindowLoss(CharFuncLoss):
    """Consecutive windows of a fixed size."""

    @classmethod
    def _build_comp_idxs(
        cls, D: int, window_size: int = 2, **kwargs
    ) -> List[Tensor]:
        return [
            torch.arange(i, i + window_size, dtype=torch.long)
            for i in range(D - window_size + 1)
        ]


__all__ = [
    "DEFAULT_MC_POINTS",
    "DEFAULT_MC_BOUND",
    "DEFAULT_KERNEL",
    "DEFAULT_RISK",
    "MonteCarloEstimatedCF",
    "CharFuncComponent",
    "CharFuncLoss",
    "CFFullLoss",
    "CFPairwiseLoss",
    "CFMarginalLoss",
    "CFLongTermPairsLoss",
    "CFRollingWindowLoss",
]
