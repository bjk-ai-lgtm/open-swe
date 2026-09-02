"""Hermetic environment defaults for orchestration tests."""

import pytest


@pytest.fixture(autouse=True)
def isolate_sandbox_provider(monkeypatch):
    """Do not inherit the developer's real sandbox provider."""
    monkeypatch.delenv(
        "SANDBOX_TYPE",
        raising=False,
    )
