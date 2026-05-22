"""Thread-limit scripts for numerical forecast runs.

These settings avoid CAMB/OpenMP thread issues that can otherwise cause
segmentation faults when CCL calls CAMB. They must be applied before importing
NumPy, SciPy, PyCCL, CAMB, or libraries that import them internally.

This is intentionally conservative: most scripts do not need many numerical
backend threads, and setting the limits early makes local forecast runs more
stable.
"""

from __future__ import annotations

import os

THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def limit_numerical_threads(n_threads: int = 1) -> None:
    """Limit numerical backend threads before CCL/CAMB are imported.

    Args:
        n_threads: Number of threads to allow for OpenMP, BLAS, and NumExpr
            backends. Defaults to 1.
    """
    value = str(int(n_threads))

    for name in THREAD_LIMIT_ENV_VARS:
        os.environ[name] = value
