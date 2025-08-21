from __future__ import annotations

from typing import Iterable, Optional, Union

import torch
from torch import LongTensor, Tensor


def _normalize_data(data: Tensor, is_batch_first: bool = False) -> Tensor:
    """Convert ``data`` to shape ``(B, D, 1)`` (batch‑first).

    Accepted input shapes:
    * ``(B, D)``            – interpreted as batch‑first, a trailing 1 is added
    * ``(B, D, 1)``        – already OK
    * ``(D, B)``            – transposed, then a trailing 1 is added
    * ``(D, B, 1)``        – transposed
    """
    if data.ndim == 2:
        # (B, D)  or (D, B)
        if is_batch_first:
            return data.unsqueeze(-1)  # (B, D, 1)
        else:
            return data.t().unsqueeze(-1)  # (B, D, 1)
    elif data.ndim == 3 and data.shape[2] == 1:
        # (B, D, 1)  or (D, B, 1)
        if is_batch_first:
            return data
        else:
            return data.transpose(0, 1)  # swap batch & time
    raise ValueError(f"data must be 2‑D (B×D) or 3‑D (B×D×1); got shape {tuple(data.shape)}")


def _lags_to_idx_tensor(lags: Union[int, Iterable[int]], device: Optional[torch.Device] = None) -> LongTensor:
    if isinstance(lags, int):
        return torch.arange(lags, device=device, dtype=torch.int32)  # type: ignore
    else:
        return torch.tensor(list(lags), device=device, dtype=torch.int32)  # type: ignore
