"""Anotaions for different Callable types used within project."""

from typing import Callable, Concatenate, TypeAlias

from torch import Tensor

ElementwiseFn: TypeAlias = Callable[Concatenate[Tensor, ...], Tensor]
"""This is a type alias for functions, which are computed on tensor per-
element."""

BatchedTensorFn: TypeAlias = Callable[Concatenate[Tensor, ...], Tensor]
"""This is a type alias for functions, which are computed on batched tensor in
the form (batch, ...)."""

LossFn = Callable[Concatenate[Tensor, Tensor, ...], Tensor]
"""
This is a type alias for functions, which are computed on
two non-batched tensors of same shape and return either 
* Tensor of the same shape as input tensors
* 0-dimensional tensor
"""
