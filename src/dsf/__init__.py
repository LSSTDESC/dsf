"""Delta Sigma forecasting tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dsf.utils.colors import (
    DSF_CMAP,
    DSF_CMAP_DISCRETE,
    DSF_COLORS,
    DSF_NORM,
)

if TYPE_CHECKING:
    from dsf.delta_sigma_forecast_builder import DeltaSigmaForecastBuilder

__all__ = [
    "DeltaSigmaForecastBuilder",
    "DSF_COLORS",
    "DSF_CMAP",
    "DSF_CMAP_DISCRETE",
    "DSF_NORM",
]


def __getattr__(name: str) -> Any:
    """Lazily import the forecasting builder when requested."""
    if name == "DeltaSigmaForecastBuilder":
        from dsf.delta_sigma_forecast_builder import DeltaSigmaForecastBuilder

        return DeltaSigmaForecastBuilder

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
