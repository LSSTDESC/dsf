"""Shared type aliases for Delta Sigma forecasts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ArrayLike",
    "BinPairs",
    "FloatArray",
    "FloatLike",
    "ScalarOrPerBin",
    "SpectrumInput",
    "ThetaMapper",
    "TomoBins",
    "TomographyInputs",
]

FloatArray: TypeAlias = NDArray[np.float64]
FloatLike: TypeAlias = float | FloatArray
ArrayLike: TypeAlias = FloatArray | list[float] | tuple[float, ...]

BinPairs: TypeAlias = list[tuple[int, int]]
TomoBins: TypeAlias = dict[int, dict[str, Any]]
TomographyInputs: TypeAlias = dict[str, Any]


ScalarOrPerBin = float | Sequence[float] | Mapping[int | str, float]

SpectrumInput: TypeAlias = ArrayLike | Callable[..., FloatArray]

ThetaMapper = Callable[[FloatArray, dict[str, Any]], Mapping[str, Any]]
