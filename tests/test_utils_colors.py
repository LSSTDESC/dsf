"""Tests for the DSF color palette utilities."""

import numpy as np
import pytest
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, to_rgb

from dsf import DSF_CMAP, DSF_CMAP_DISCRETE, DSF_COLORS, DSF_NORM
from dsf.utils.colors import PASTEL_AMOUNT, pastelize

BASE_COLORS = [
    "#294653",
    "#168C83",
    "#68C8B8",
    "#F2C75C",
    "#F6A35C",
    "#ED765E",
    "#D95C48",
]


@pytest.mark.parametrize(
    ("color", "amount", "expected"),
    [
        ("#000000", 0.0, "#000000"),
        ("#000000", 1.0, "#ffffff"),
        ("#000000", 0.5, "#808080"),
        ("#ffffff", 0.5, "#ffffff"),
        ("#ff0000", 0.5, "#ff8080"),
        ("#0000ff", 0.5, "#8080ff"),
    ],
)
def test_pastelize(color, amount, expected):
    """Test pastelization against known hexadecimal values."""
    assert pastelize(color, amount) == expected


@pytest.mark.parametrize("amount", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_pastelize_interpolation(amount):
    """Test that pastelization linearly interpolates toward white."""
    color = "#294653"
    expected = (1.0 - amount) * np.array(to_rgb(color)) + amount

    np.testing.assert_allclose(to_rgb(pastelize(color, amount)), expected, atol=1 / 255)


@pytest.mark.parametrize("color", BASE_COLORS)
def test_pastelize_endpoints(color):
    """Test that pastelization preserves colors at zero and produces white at one."""
    assert pastelize(color, 0.0) == color.lower()
    assert pastelize(color, 1.0) == "#ffffff"


def test_dsf_colors_length():
    """Test that the DSF palette contains seven colors."""
    assert len(DSF_COLORS) == 7


def test_dsf_colors_hex():
    """Test that all DSF colors are valid hexadecimal strings."""
    for color in DSF_COLORS:
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7
        assert all(char in "0123456789abcdef" for char in color[1:])


def test_dsf_colors_unique():
    """Test that the DSF palette contains no duplicate colors."""
    assert len(set(DSF_COLORS)) == len(DSF_COLORS)


def test_dsf_colors_values():
    """Test that the DSF palette matches the pastelized base colors."""
    expected = [pastelize(color, PASTEL_AMOUNT) for color in BASE_COLORS]

    assert DSF_COLORS == expected


def test_dsf_cmap_type():
    """Test that the continuous DSF colormap has the correct type."""
    assert isinstance(DSF_CMAP, LinearSegmentedColormap)


def test_dsf_cmap_name():
    """Test that the continuous DSF colormap has the correct name."""
    assert DSF_CMAP.name == "dsf_colors"


def test_dsf_cmap_resolution():
    """Test that the continuous DSF colormap has 256 levels."""
    assert DSF_CMAP.N == 256


def test_dsf_cmap_endpoints():
    """Test that the continuous colormap preserves the palette endpoints."""
    np.testing.assert_allclose(DSF_CMAP(0.0)[:3], to_rgb(DSF_COLORS[0]))
    np.testing.assert_allclose(DSF_CMAP(1.0)[:3], to_rgb(DSF_COLORS[-1]))


def test_dsf_cmap_discrete_type():
    """Test that the discrete DSF colormap has the correct type."""
    assert isinstance(DSF_CMAP_DISCRETE, ListedColormap)


def test_dsf_cmap_discrete_name():
    """Test that the discrete DSF colormap has the correct name."""
    assert DSF_CMAP_DISCRETE.name == "dsf_colors_discrete"


def test_dsf_cmap_discrete_resolution():
    """Test that the discrete DSF colormap contains seven levels."""
    assert DSF_CMAP_DISCRETE.N == len(DSF_COLORS)


def test_dsf_cmap_discrete_colors():
    """Test that the discrete colormap reproduces the DSF palette."""
    for i, color in enumerate(DSF_COLORS):
        np.testing.assert_allclose(DSF_CMAP_DISCRETE(i)[:3], to_rgb(color))


def test_dsf_norm_type():
    """Test that the DSF normalization uses discrete boundaries."""
    assert isinstance(DSF_NORM, BoundaryNorm)


def test_dsf_norm_boundaries():
    """Test that the normalization boundaries cover all seven colors."""
    expected = np.arange(len(DSF_COLORS) + 1)

    np.testing.assert_array_equal(DSF_NORM.boundaries, expected)


def test_dsf_norm_indices():
    """Test that category centers map to their corresponding color indices."""
    values = np.arange(len(DSF_COLORS)) + 0.5
    expected = np.arange(len(DSF_COLORS))

    np.testing.assert_array_equal(DSF_NORM(values), expected)


def test_dsf_norm_colormap():
    """Test that normalization selects the expected discrete colors."""
    for i, color in enumerate(DSF_COLORS):
        index = DSF_NORM(i + 0.5)
        np.testing.assert_allclose(DSF_CMAP_DISCRETE(index)[:3], to_rgb(color))


def test_dsf_colors_top_level_import():
    """Test that the DSF palette is exposed through the package API."""
    from dsf.utils.colors import DSF_CMAP as cmap
    from dsf.utils.colors import DSF_CMAP_DISCRETE as cmap_discrete
    from dsf.utils.colors import DSF_COLORS as colors
    from dsf.utils.colors import DSF_NORM as norm

    assert DSF_COLORS is colors
    assert DSF_CMAP is cmap
    assert DSF_CMAP_DISCRETE is cmap_discrete
    assert DSF_NORM is norm
