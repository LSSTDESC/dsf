"""Tests for ``src.dsf.utils.types.py``."""

from collections.abc import Callable, Mapping, Sequence
from typing import get_args, get_origin

import numpy as np

from src.dsf.utils.types import (
    ArrayLike,
    BinPairs,
    FloatArray,
    FloatLike,
    ScalarOrPerBin,
    SpectrumInput,
    ThetaMapper,
    TomoBins,
    TomographyInputs,
)


def test_float_array_alias_points_to_numpy_float64_array():
    """Tests that FloatArray aliases NumPy float64 arrays."""
    assert get_origin(FloatArray) is np.ndarray


def test_float_like_accepts_float_and_float_array_aliases():
    """Tests that FloatLike includes scalar floats and FloatArray."""
    assert float in get_args(FloatLike)
    assert FloatArray in get_args(FloatLike)


def test_array_like_accepts_arrays_lists_and_tuples():
    """Tests that ArrayLike includes arrays, lists, and tuples."""
    args = get_args(ArrayLike)

    assert FloatArray in args
    assert list[float] in args
    assert tuple[float, ...] in args


def test_bin_pairs_alias_is_list_of_integer_pairs():
    """Tests that BinPairs aliases a list of integer index pairs."""
    assert get_origin(BinPairs) is list
    assert get_args(BinPairs) == (tuple[int, int],)


def test_tomo_bins_alias_is_dictionary_of_bin_metadata():
    """Tests that TomoBins aliases integer-keyed bin metadata dictionaries."""
    assert get_origin(TomoBins) is dict
    assert get_args(TomoBins)[0] is int


def test_tomography_inputs_alias_is_string_keyed_dictionary():
    """Tests that TomographyInputs aliases a string-keyed dictionary."""
    assert get_origin(TomographyInputs) is dict
    assert get_args(TomographyInputs)[0] is str


def test_scalar_or_per_bin_accepts_scalar_sequence_and_mapping():
    """Tests that ScalarOrPerBin allows scalar and per-bin values."""
    args = get_args(ScalarOrPerBin)

    assert float in args
    assert Sequence[float] in args
    assert Mapping[int | str, float] in args


def test_spectrum_input_accepts_array_like_members_or_callable():
    """Tests that SpectrumInput allows tabulated arrays or callables."""
    args = get_args(SpectrumInput)

    assert FloatArray in args
    assert list[float] in args
    assert tuple[float, ...] in args
    assert Callable[..., FloatArray] in args


def test_theta_mapper_alias_is_callable_returning_mapping():
    """Tests that ThetaMapper aliases a callable parameter mapper."""
    assert get_origin(ThetaMapper) is Callable
