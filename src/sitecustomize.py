"""Project-level Python startup settings for stable numerical forecast runs.

Python imports this module automatically at interpreter startup when the
repository's ``src`` directory is importable, for example after installing the
project in editable mode with ``pip install -e .``.

The purpose of this file is to apply conservative numerical thread limits
before libraries such as NumPy, SciPy, PyCCL, or CAMB are imported. This helps
avoid CAMB/OpenMP thread issues that can otherwise appear when forecast scripts
or notebooks initialize the numerical stack.

The defaults can be changed from the shell with ``DSF_NUM_THREADS``. The startup
limits can be disabled entirely with ``DSF_DISABLE_THREAD_LIMITS=1``.
"""

from __future__ import annotations

import os

from src.dsf.utils.thread_limits import limit_numerical_threads

if os.environ.get("DSF_DISABLE_THREAD_LIMITS") != "1":
    n_threads = int(os.environ.get("DSF_NUM_THREADS", "1"))
    limit_numerical_threads(n_threads)
