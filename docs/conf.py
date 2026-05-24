"""Sphinx configuration for DSF documentation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "DSF"
author = "Nikolina Sarcevic, Ben Levine"
copyright = "2026, LSST Dark Energy Science Collaboration"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "api/modules.rst"]

html_theme = "furo"
html_static_path = ["_static"]

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False