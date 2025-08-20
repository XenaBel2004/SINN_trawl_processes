from __future__ import annotations

from typing import Iterable, Literal, Optional, Union

import torch
from torch import LongTensor, Tensor

from .normalization import _lags_to_idx_tensor, _normalize_data


class ACF:
    def __init__(
        self,
        lags: Optional[Union[int, Iterable[int], LongTensor]] = None,
        method: Literal["fft", "brute"] = "fft",
        data_batch_first: bool = False,
    ):
        self.lags = lags
        self.method = method
        self.data_batch_first = data_batch_first

    @staticmethod
    def via_fft(x: Tensor, lags: LongTensor) -> Tensor:
        """Autocorrelation function using the FFT trick.

        Parameters
        ----------
        x : Tensor
            Shape ``(D, B, 1)``.
        lags : int
            Number of lags to return (``0 … lags-1``).

        Returns
        -------
        Tensor
            Shape ``(lags, ...)`` – normalised autocorrelation for each batch /
            variable pair.

        """
        # Demean the time series
        x_centered = x[:, :, 0] - x[:, :, 0].mean()

        # Zero‑padding to avoid circular convolution artefacts
        n = x_centered.shape[1]
        n_fft = n * 2 - 1
        print(lags)
        # FFT, multiply by complex conjugate, inverse FFT → autocovariance
        f = torch.fft.fft(x_centered, n=n_fft, dim=1)
        acov = torch.fft.ifft(f * f.conj(), dim=1).real[:, :n].mean(dim=0)

        # Normalise – the zero‑lag term is the variance
        acf = acov[lags] / acov[0]
        return acf

    @staticmethod
    def via_bruteforce(x: Tensor, lags: LongTensor) -> Tensor:
        """Direct (O(N·L)) autocorrelation for an arbitrary list of lags.

        Parameters
        ----------
        x : Tensor
            Shape ``(D, B, ...)``.
        lags : int or iterable of int
            If an int, compute ``0 … lags-1``; otherwise a custom list.

        Returns
        -------
        Tensor
            Shape ``(len(lags))``.

        """
        # Demean the time series
        x_centered = x[:, :, 0] - x[:, :, 0].mean()

        out = torch.empty(lags.shape[0], dtype=x.dtype, device=x.device)

        for i, lag in enumerate(lags):
            if lag == 0:
                out[i] = 1.0
            else:
                # Overlap the two shifted series
                u = x_centered[:, :-lag]
                v = x_centered[:, lag:]
                out[i] = torch.sum(u * v, dim=[0, 1]) / torch.sqrt(
                    torch.sum(torch.square(u), dim=[0, 1]) * torch.sum(torch.square(v), dim=[0, 1])
                )
        return out

    def __call__(self, x: Tensor) -> Tensor:
        x = _normalize_data(x, self.data_batch_first)

        if self.lags is None:
            lags = _lags_to_idx_tensor(x.shape[1], device=x.device)
        elif not isinstance(self.lags, LongTensor):
            lags = _lags_to_idx_tensor(self.lags, device=x.device)
        else:
            lags = self.lags.to(x.device)  # type: ignore

        if self.method == "fft":
            return ACF.via_fft(x, lags)
        if self.method == "brute":
            return ACF.via_bruteforce(x, lags)

        raise NotImplementedError
