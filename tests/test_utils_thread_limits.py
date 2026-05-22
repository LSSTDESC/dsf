"""Tests for ``src.dsf.utils.thread_limits.py``."""

import os

from src.dsf.utils.thread_limits import THREAD_LIMIT_ENV_VARS, limit_numerical_threads


def test_thread_limit_env_vars_lists_expected_backend_variables():
    """Tests that thread-limit variables include common numerical backends."""
    assert THREAD_LIMIT_ENV_VARS == (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )


def test_limit_numerical_threads_sets_all_backend_variables(monkeypatch):
    """Tests that all numerical backend thread variables are set."""
    for name in THREAD_LIMIT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    limit_numerical_threads(2)

    for name in THREAD_LIMIT_ENV_VARS:
        assert os.environ[name] == "2"


def test_limit_numerical_threads_defaults_to_one_thread(monkeypatch):
    """Tests that the default thread limit is one."""
    for name in THREAD_LIMIT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    limit_numerical_threads()

    for name in THREAD_LIMIT_ENV_VARS:
        assert os.environ[name] == "1"


def test_limit_numerical_threads_casts_input_to_integer_string(monkeypatch):
    """Tests that thread limits are stored as integer strings."""
    for name in THREAD_LIMIT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    limit_numerical_threads(3.9)

    for name in THREAD_LIMIT_ENV_VARS:
        assert os.environ[name] == "3"


def test_limit_numerical_threads_overwrites_existing_values(monkeypatch):
    """Tests that existing backend thread limits are overwritten."""
    for name in THREAD_LIMIT_ENV_VARS:
        monkeypatch.setenv(name, "8")

    limit_numerical_threads(1)

    for name in THREAD_LIMIT_ENV_VARS:
        assert os.environ[name] == "1"
