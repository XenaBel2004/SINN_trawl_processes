from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal

from .noise import NoiseProcess


class GaussianNoise(NoiseProcess):
    def __init__(self, mean: float = 0.0, var: float = 1.0, **default_opts) -> None:
        self.mean: Tensor = torch.tensor(mean, device=default_opts.get("device"))
        self.var: Tensor = torch.tensor(var, device=default_opts.get("device"))
        distr: Distribution = Normal(self.mean, self.var**0.5)

        def cumulant(u: Tensor) -> Tensor:
            return torch.exp(1.0j * self.mean * u - self.var * u**2 / 2)

        super().__init__(distr, cumulant, **default_opts)


class ExponentialNoise(NoiseProcess):
    def __init__(self, rate: float = 1.0, **default_opts) -> None:
        self.rate: Tensor = torch.tensor(rate, device=default_opts.get("device"))
        distr: Distribution = Gamma(torch.tensor(1.0), self.rate)

        def cumulant(u: Tensor) -> Tensor:
            return -torch.log(1 - 1.0j * u / self.rate)

        super().__init__(distr, cumulant, **default_opts)
