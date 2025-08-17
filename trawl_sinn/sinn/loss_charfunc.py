# ──────────────────────────────────────────────────────────────────────
# charfunc_losses.py
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from .base_loss import BaseStatLoss
from ..processes import StationaryStochasticProcess
from tqdm.auto import tqdm

# --------------------------------------------------------------------- #
# Module‑level defaults (centralised constants)
# --------------------------------------------------------------------- #
DEFAULT_MC_POINTS: int = 1_000  # number of Monte‑Carlo draws
DEFAULT_MC_BOUND: float = 1.0  # integration bound for uniform draw
DEFAULT_KERNEL: Callable[[Tensor], Tensor] = lambda theta: torch.ones(
    theta.shape[0], device=theta.device
)  # importance‑sampling kernel
DEFAULT_RISK: Callable[[Tensor, Tensor], Tensor] = lambda a, b: torch.abs(
    a - b
)  # point‑wise loss (|·|)


# --------------------------------------------------------------------- #
# Helper – MC estimator of a (possibly weighted) characteristic function
# --------------------------------------------------------------------- #
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
        data_batched_first: Tensor,
        kernel: Callable[[Tensor], Tensor] = DEFAULT_KERNEL,
    ) -> None:
        self.data = data_batched_first  # (B, D)
        self.kernel = kernel

    def __call__(self, theta: Tensor) -> Tensor:
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
        kern = self.kernel(theta)  # (M,)
        # inner product for each MC point: (B, D) @ (D, M) → (B, M)
        cumulant = 1.0j * (self.data @ theta.t())  # (B, M)
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
        shape ``(D, M)`` and return a complex tensor of shape ``(M,)``.
    """

    idx: Tensor
    target_fn: Callable[[Tensor], Tensor]

    def _mc_estimate(
        self,
        data_batched_first: Tensor,
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
        device = data_batched_first.device
        _, D, _ = data_batched_first.shape

        # Random MC points uniformly drawn from [-bound, bound]^D
        theta = (
            2.0 * mc_bound * torch.rand(mc_points, self.idx.shape[0], device=device)
            - mc_bound
        )

        # Slice the data for the selected indices
        data_slice = data_batched_first[:, self.idx, 0]  # (B, L)
        empirical_cf = MonteCarloEstimatedCF(data_slice, data_kernel)

        # Evaluate both empirical and target CFs, compute the risk and average
        loss = risk(empirical_cf(theta), self.target_fn(theta)).mean()
        return loss


# --------------------------------------------------------------------- #
# Composite loss – weighted sum of components
# --------------------------------------------------------------------- #
class CharFuncLoss(BaseStatLoss):
    """
    Weighted sum of :class:`CharFuncComponent` objects.

    The class knows how to turn user data into the required batch‑first format,
    runs each component's MC estimator and (optionally) applies a Jacobian
    normalisation factor.
    """

    def __init__(
        self,
        components: Sequence[CharFuncComponent],
        *,
        data_kernel: Callable[[Tensor], Tensor] = DEFAULT_KERNEL,
        risk: Callable[[Tensor, Tensor], Tensor] = DEFAULT_RISK,
        mc_points: int = DEFAULT_MC_POINTS,
        mc_bound: float = DEFAULT_MC_BOUND,
        dim_normalization: bool = True,
        data_batch_first: bool = True,
    ) -> None:
        super().__init__(torch.tensor(0.0), lambda t: torch.tensor(0.0))  # type: ignore
        if not components:
            raise ValueError("CharFuncLoss needs at least one component")
        self.components = list(components)
        self.dim_normalization = dim_normalization
        self.data_batch_first_default = data_batch_first
        # Store defaults for the MC estimator
        self.data_kernel = data_kernel
        self.risk = risk
        self.mc_points = mc_points
        self.mc_bound = mc_bound

    # -----------------------------------------------------------------
    # 1. Normalise the user supplied data to (B, D, 1) batch‑first
    # -----------------------------------------------------------------
    @staticmethod
    def _as_batch_first(data: Tensor, data_batch_first: bool = False) -> Tensor:
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
            if data_batch_first:
                out = data.unsqueeze(-1)  # (B, D, 1)
            else:
                out = data.t().unsqueeze(-1)  # (B, D, 1)
        elif data.ndim == 3:
            # (B, D, 1)  or (D, B, 1)
            if data_batch_first:
                out = data
            else:
                out = data.transpose(0, 1)  # swap batch & time
        else:
            raise ValueError(
                f"data must be 2‑D (B×D) or 3‑D (B×D×1); got shape {tuple(data.shape)}"
            )
        return out.contiguous()

    # -----------------------------------------------------------------
    # 2. Forward – sum over components
    # -----------------------------------------------------------------
    def forward(
        self,
        data: Tensor,
        *,
        data_batch_first: Optional[bool] = None,
        disable_tqdm: bool = True,
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
        data_prepped = self._as_batch_first(data, data_batch_first)
        loss = torch.tensor(0.0, device=data_prepped.device, dtype=data_prepped.dtype)
        for comp in tqdm(self.components, disable=disable_tqdm):
            term = comp._mc_estimate(
                data_prepped,
                data_kernel=self.data_kernel,
                mc_points=self.mc_points,
                mc_bound=self.mc_bound,
                risk=self.risk,
            )
            if self.dim_normalization:
                # Jacobian factor (2·bound)^{|idx|}
                term = ((2.0 * self.mc_bound) ** comp.idx.numel()) * term
            loss = loss + term
        return loss


# --------------------------------------------------------------------- #
# Helper – create a component from a model FDD or from empirical data
# --------------------------------------------------------------------- #
def _make_cf_component_from_fdd(
    idx: Tensor,
    times: Tensor,
    process: "StationaryStochasticProcess",
) -> CharFuncComponent:
    """
    Build a :class:`CharFuncComponent` from a model's finite‑dimensional
    distribution.

    Parameters
    ----------
    idx : Tensor
        1‑D integer tensor of selected time indices.
    times : Tensor
        Full vector of observation times.
    process : StationaryStochasticProcess
        The stochastic process model; ``process.at_times(times[idx])`` must
        return an object exposing a ``charfunc`` method.

    Returns
    -------
    CharFuncComponent
    """
    fdd = process.at_times(times[idx])
    return CharFuncComponent(
        idx=idx, target_fn=lambda theta: fdd.charfunc(theta, theta_batch_first=True)
    )


def _make_cf_component_from_data(
    idx: Tensor,
    sampled_data: Tensor,
    *,
    kernel: Callable[[Tensor], Tensor] = DEFAULT_KERNEL,
) -> CharFuncComponent:
    """
    Build a :class:`CharFuncComponent` from observed data.

    Parameters
    ----------
    idx : Tensor
        1‑D integer tensor of selected time indices.
    sampled_data : Tensor
        Observation matrix, shape ``(B, D)`` or ``(B, D, 1)`` (batch‑first).
    kernel : Callable, optional
        Importance‑sampling kernel for the empirical characteristic function.

    Returns
    -------
    CharFuncComponent
    """
    # Ensure data are in batch‑first (B, D, 1) layout
    data_batched = CharFuncLoss._as_batch_first(sampled_data, data_batch_first=True)
    target_fn = MonteCarloEstimatedCF(data_batched[:, idx, 0], kernel)
    return CharFuncComponent(idx=idx, target_fn=target_fn)


# --------------------------------------------------------------------- #
# Concrete loss subclasses – each implements a specific “scheme” and offers
# factory classmethods ``from_fdd`` and ``from_empirical``.
# --------------------------------------------------------------------- #
class CFFullLoss(CharFuncLoss):
    """Single component that uses *all* observation times."""

    @classmethod
    def from_fdd(
        cls,
        process: "StationaryStochasticProcess",
        times: Tensor,
        **loss_kwargs,
    ) -> "CFFullLoss":
        idx = torch.arange(times.shape[0], dtype=torch.long, device=times.device)
        comp = _make_cf_component_from_fdd(idx, times, process)
        return cls([comp], **loss_kwargs)

    @classmethod
    def from_empirical(
        cls,
        sampled_data: Tensor,
        times: Tensor,
        **loss_kwargs,
    ) -> "CFFullLoss":
        idx = torch.arange(times.shape[0], dtype=torch.long, device=times.device)
        comp = _make_cf_component_from_data(
            idx, times, sampled_data, kernel=DEFAULT_KERNEL
        )
        return cls([comp], **loss_kwargs)


class CFPairwiseLoss(CharFuncLoss):
    """All unordered pairs of observation times (i < j)."""

    @classmethod
    def _pairwise_indices(cls, n: int, device: torch.device) -> List[Tensor]:
        """Utility: return a list of ``torch.tensor([i, j])`` for i<j."""
        return [
            torch.tensor([i, j], dtype=torch.long, device=device)
            for i in range(n - 1)
            for j in range(i + 1, n)
        ]

    @classmethod
    def from_fdd(
        cls,
        process: "StationaryStochasticProcess",
        times: Tensor,
        **loss_kwargs,
    ) -> "CFPairwiseLoss":
        n = times.shape[0]
        comps = [
            _make_cf_component_from_fdd(idx, times, process)
            for idx in cls._pairwise_indices(n, times.device)
        ]
        return cls(comps, **loss_kwargs)

    @classmethod
    def from_empirical(
        cls,
        sampled_data: Tensor,
        times: Tensor,
        **loss_kwargs,
    ) -> "CFPairwiseLoss":
        n = times.shape[0]
        comps = [
            _make_cf_component_from_data(
                idx, times, sampled_data, kernel=DEFAULT_KERNEL
            )
            for idx in cls._pairwise_indices(n, times.device)
        ]
        return cls(comps, **loss_kwargs)


class CFMarginalLoss(CharFuncLoss):
    """One component per single observation time (marginals)."""

    @classmethod
    def from_fdd(
        cls,
        process: "StationaryStochasticProcess",
        times: Tensor,
        **loss_kwargs,
    ) -> "CFMarginalLoss":
        comps = [
            _make_cf_component_from_fdd(
                torch.tensor([i], dtype=torch.long, device=times.device), times, process
            )
            for i in range(times.shape[0])
        ]
        return cls(comps, **loss_kwargs)

    @classmethod
    def from_empirical(
        cls,
        sampled_data: Tensor,
        **loss_kwargs,
    ) -> "CFMarginalLoss":
        comps = [
            _make_cf_component_from_data(
                torch.tensor([i], dtype=torch.long, device=times.device),
                sampled_data,
                kernel=DEFAULT_KERNEL,
            )
            for i in range(times.shape[0])
        ]
        return cls(comps, **loss_kwargs)


class CFLongTermPairsLoss(CharFuncLoss):
    """
    Sparse “long‑term’’ pairs scheme: from each start index ``i`` (stride ``step``)
    pair it with ``j = i + d, i + 2·d, …``.
    """

    @classmethod
    def _long_term_pairs(
        cls,
        n: int,
        step: int,
        d: int,
        device: torch.device,
    ) -> List[Tensor]:
        pairs: List[Tensor] = []
        for i in range(0, n - 1, step):
            for j in range(i + d, n, d):
                pairs.append(torch.tensor([i, j], dtype=torch.long, device=device))
        return pairs

    @classmethod
    def from_fdd(
        cls,
        process: "StationaryStochasticProcess",
        times: Tensor,
        *,
        step: int = 1,
        d: int = 10,
        **loss_kwargs,
    ) -> "CFLongTermPairsLoss":
        n = times.shape[0]
        comps = [
            _make_cf_component_from_fdd(idx, times, process)
            for idx in cls._long_term_pairs(n, step, d, times.device)
        ]
        return cls(comps, **loss_kwargs)

    @classmethod
    def from_empirical(
        cls,
        sampled_data: Tensor,
        times: Tensor,
        *,
        step: int = 1,
        d: int = 10,
        **loss_kwargs,
    ) -> "CFLongTermPairsLoss":
        n = times.shape[0]
        comps = [
            _make_cf_component_from_data(
                idx, times, sampled_data, kernel=DEFAULT_KERNEL
            )
            for idx in cls._long_term_pairs(n, step, d, times.device)
        ]
        return cls(comps, **loss_kwargs)


class CFRollingWindowLoss(CharFuncLoss):
    """Consecutive windows of a fixed size."""

    @classmethod
    def _windows(
        cls,
        n: int,
        window_size: int,
        device: torch.device,
    ) -> List[Tensor]:
        return [
            torch.arange(i, i + window_size, dtype=torch.long, device=device)
            for i in range(n - window_size + 1)
        ]

    @classmethod
    def from_fdd(
        cls,
        process: "StationaryStochasticProcess",
        times: Tensor,
        *,
        window_size: int = 2,
        **loss_kwargs,
    ) -> "CFRollingWindowLoss":
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        n = times.shape[0]
        comps = [
            _make_cf_component_from_fdd(idx, times, process)
            for idx in cls._windows(n, window_size, times.device)
        ]
        return cls(comps, **loss_kwargs)

    @classmethod
    def from_empirical(
        cls,
        sampled_data: Tensor,
        times: Tensor,
        *,
        window_size: int = 2,
        **loss_kwargs,
    ) -> "CFRollingWindowLoss":
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        n = times.shape[0]
        comps = [
            _make_cf_component_from_data(
                idx, times, sampled_data, kernel=DEFAULT_KERNEL
            )
            for idx in cls._windows(n, window_size, times.device)
        ]
        return cls(comps, **loss_kwargs)


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
