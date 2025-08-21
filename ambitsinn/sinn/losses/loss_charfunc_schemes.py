from typing import List

from .loss_charfunc_base import CharFuncLoss


# --------------------------------------------------------------------- #
# Concrete loss subclasses - each implements a specific scheme and offers
# factory classmethods ``from_fdd`` and ``from_empirical``.
# --------------------------------------------------------------------- #
class CFFullLoss(CharFuncLoss):
    """Single component that uses *all* observation times."""

    @classmethod
    def _build_comp_idxs(cls, D: int, **kwargs) -> List[List[int]]:
        return [list(range(D))]


class CFPairwiseLoss(CharFuncLoss):
    """All unordered pairs of observation times (i < j)."""

    @classmethod
    def _build_comp_idxs(cls, D: int, **kwargs) -> List[List[int]]:
        return [[i, j] for i in range(D - 1) for j in range(i + 1, D)]


class CFMarginalLoss(CharFuncLoss):
    """One component per single observation time (marginals)."""

    @classmethod
    def _build_comp_idxs(cls, D: int, **kwargs) -> List[List[int]]:
        return [[i] for i in range(D)]


class CFLongTermPairsLoss(CharFuncLoss):
    """Sparse long-term'' pairs scheme: from each start index ``i`` (stride
    ``step``) pair it with ``j = i + d, i + 2·d, ...``.
    """

    @classmethod
    def _build_comp_idxs(cls, D: int, roll_step: int = 10, pair_step: int = 10, **kwargs) -> List[List[int]]:
        pairs: List[List[int]] = []
        for i in range(0, D - 1, roll_step):
            for j in range(i + pair_step, D, pair_step):
                pairs.append([i, j])
        return pairs


class CFRollingWindowLoss(CharFuncLoss):
    """Consecutive windows of a fixed size."""

    @classmethod
    def _build_comp_idxs(cls, D: int, window_size: int = 2, **kwargs) -> List[List[int]]:
        return [list(range(i, i + window_size)) for i in range(D - window_size + 1)]
