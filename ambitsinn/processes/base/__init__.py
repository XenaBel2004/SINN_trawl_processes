# -*- coding: ascii -*-
"""Base classes that define the public API for stationary stochastic processes and their
finite-dimensional distributions (FDDs).

The module provides:
* ``StationaryStochasticProcess`` - an abstract process class with validation,
  cumulant/characteristic-function helpers and a generic sampling entry point.
* ``StationaryProcessFDD`` - an abstract FDD class that knows the observation
  grid and implements the heavy-lifting for cumulants, characteristic functions
  and sampling.
"""

from .stationary_process_fdd import StationaryProcessFDD as StationaryProcessFDD
from .stationary_stochastic_process import (
    StationaryStochasticProcess as StationaryStochasticProcess,
)

__all__ = [
    "StationaryStochasticProcess",
    "StationaryProcessFDD",
]
