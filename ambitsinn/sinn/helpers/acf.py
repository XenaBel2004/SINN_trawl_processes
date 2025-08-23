from __future__ import annotations

from typing import Iterable, List, Literal, Optional, Union

from torch import IntTensor, LongTensor, Tensor

from .normalization import _lags_to_idx_tensor, _normalize_data
from .statistics import acf_bruteforce, acf_fft


class ACF:
    def __init__(
        self,
        lags: Optional[Union[int, Iterable[int], IntTensor, LongTensor]] = None,
        method: Literal["fft", "brute"] = "fft",
        data_batch_first: bool = False,
    ):
        self.lags = lags
        self.method = method
        self.data_batch_first = data_batch_first

    def __call__(self, x: Tensor) -> Tensor:
        x = _normalize_data(x, self.data_batch_first)

        if self.lags is None:
            lags = _lags_to_idx_tensor(x.shape[1])
        elif not isinstance(self.lags, List):
            lags = _lags_to_idx_tensor(self.lags)
        else:
            lags = self.lags

        if self.method == "fft":
            return acf_fft(x, lags)
        if self.method == "brute":
            return acf_bruteforce(x, lags)

        raise NotImplementedError
