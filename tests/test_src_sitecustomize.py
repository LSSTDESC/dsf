"""Unit tests for project-level sitecustomize startup settings."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SITECUSTOMIZE_PATH = Path(__file__).resolve().parents[1] / "src" / "sitecustomize.py"


def import_sitecustomize_from_path(monkeypatch, fake_limit):
    """Import src/sitecustomize.py after patching the thread-limit dependency."""
    import dsf.utils.thread_limits as thread_limits

    monkeypatch.setattr(thread_limits, "limit_numerical_threads", fake_limit)

    spec = importlib.util.spec_from_file_location(
        "dsf_test_sitecustomize",
        SITECUSTOMIZE_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_sitecustomize_limits_threads_to_default(monkeypatch):
    """Tests that numerical threads are limited to one by default."""
    calls = []

    def fake_limit_numerical_threads(n_threads):
        calls.append(n_threads)

    monkeypatch.delenv("DSF_DISABLE_THREAD_LIMITS", raising=False)
    monkeypatch.delenv("DSF_NUM_THREADS", raising=False)

    import_sitecustomize_from_path(monkeypatch, fake_limit_numerical_threads)

    assert calls == [1]


def test_sitecustomize_uses_configured_thread_count(monkeypatch):
    """Tests that DSF_NUM_THREADS controls the startup thread limit."""
    calls = []

    def fake_limit_numerical_threads(n_threads):
        calls.append(n_threads)

    monkeypatch.delenv("DSF_DISABLE_THREAD_LIMITS", raising=False)
    monkeypatch.setenv("DSF_NUM_THREADS", "4")

    import_sitecustomize_from_path(monkeypatch, fake_limit_numerical_threads)

    assert calls == [4]


def test_sitecustomize_can_disable_thread_limits(monkeypatch):
    """Tests that DSF_DISABLE_THREAD_LIMITS skips startup thread limiting."""
    calls = []

    def fake_limit_numerical_threads(n_threads):
        calls.append(n_threads)

    monkeypatch.setenv("DSF_DISABLE_THREAD_LIMITS", "1")
    monkeypatch.setenv("DSF_NUM_THREADS", "8")

    import_sitecustomize_from_path(monkeypatch, fake_limit_numerical_threads)

    assert calls == []


def test_sitecustomize_rejects_non_integer_thread_count(monkeypatch):
    """Tests that invalid DSF_NUM_THREADS values fail clearly on import."""
    calls = []

    def fake_limit_numerical_threads(n_threads):
        calls.append(n_threads)

    monkeypatch.delenv("DSF_DISABLE_THREAD_LIMITS", raising=False)
    monkeypatch.setenv("DSF_NUM_THREADS", "many")

    with pytest.raises(ValueError):
        import_sitecustomize_from_path(monkeypatch, fake_limit_numerical_threads)

    assert calls == []
