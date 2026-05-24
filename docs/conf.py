"""Sphinx configuration for DSF documentation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "DSF"
author = "Nikolina Sarcevic, Ben Levine"
copyright = "2026, LSST Dark Energy Science Collaboration"

html_title = "DSF Documentation"
html_baseurl = "https://lsstdesc.org/dsf/"
root_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "sphinx_design",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "api/modules.rst"]

html_theme = "furo"
html_static_path = ["_static"]

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

plot_formats = [("png", 100)]
plot_html_show_source_link = False
plot_html_show_formats = False

plot_pre_code = """
import matplotlib
matplotlib.use("Agg")
"""

html_theme_options = {
    "source_repository": "https://github.com/LSSTDESC/dsf/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/LSSTDESC/dsf",
            "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0"
                 viewBox="0 0 16 16">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53
                5.47 7.59.4.07.55-.17.55-.38
                0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94
                -.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53
                .63-.01 1.08.58 1.23.82.72 1.21 1.87.87
                2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89
                -3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02
                .08-2.12 0 0 .67-.21 2.2.82A7.65 7.65 0 0 1
                8 3.87c.68 0 1.36.09 2 .26
                1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
                .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65
                3.95.29.25.54.73.54 1.48
                0 1.07-.01 1.93-.01 2.2
                0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8
                c0-4.42-3.58-8-8-8z"></path>
            </svg>
            """,
            "class": "",
        },
    ],
}