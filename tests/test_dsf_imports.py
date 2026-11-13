"""Tests for DSF package imports."""

import subprocess
import sys


def test_dsf_import_does_not_load_forecast_builder():
    """Tests that importing DSF does not eagerly load the forecast builder."""
    code = "import dsf, sys; assert 'dsf.delta_sigma_forecast_builder' not in sys.modules"

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
