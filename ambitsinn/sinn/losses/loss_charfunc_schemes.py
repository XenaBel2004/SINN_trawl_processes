from typing import List

import torch
from torch import LongTensor

from .loss_charfunc_base import CharFuncLoss


# --------------------------------------------------------------------- #
# Concrete loss subclasses - each implements a specific scheme and offers
# factory classmethods ``from_fdd`` and ``from_empirical``.
# --------------------------------------------------------------------- #
class CFFullLoss(CharFuncLoss):
    """Single component that uses *all* observation times."""

    @classmethod
    def _build_comp_idxs(cls, D: int, **kwargs) -> List[LongTensor]:
        return torch.arange(D)  # type: ignore


class CFPairwiseLoss(CharFuncLoss):
    """All unordered pairs of observation times (i < j)."""

    @classmethod
    def _build_comp_idxs(cls, D: int, **kwargs) -> List[LongTensor]:
        return [torch.tensor([i, j]) for i in range(D - 1) for j in range(i + 1, D)]  # type: ignore


class CFMarginalLoss(CharFuncLoss):
    """One component per single observation time (marginals)."""

    @classmethod
    def _build_comp_idxs(cls, D: int, **kwargs) -> List[LongTensor]:
        return [torch.tensor([i]) for i in range(D)]  # type: ignore


class CFLongTermPairsLoss(CharFuncLoss):
    """Sparse long-term'' pairs scheme: from each start index ``i`` (stride
    ``step``) pair it with ``j = i + d, i + 2·d, ...``.
    """

    @classmethod
    def _build_comp_idxs(cls, D: int, roll_step: int = 10, pair_step: int = 10, **kwargs) -> List[LongTensor]:
        pairs: List[LongTensor] = []
        for i in range(0, D - 1, roll_step):
            for j in range(i + pair_step, D, pair_step):
                pairs.append(torch.tensor([i, j]))  # type: ignore
        return pairs


class CFRollingWindowLoss(CharFuncLoss):
    """Consecutive windows of a fixed size."""

    @classmethod
    def _build_comp_idxs(cls, D: int, window_size: int = 2, **kwargs) -> List[LongTensor]:
        return [
            torch.arange(i, i + window_size, dtype=torch.long)  # type: ignore
            for i in range(D - window_size + 1)
        ]
