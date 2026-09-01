"""Execution environment safety policy for autonomous runs."""

import os

ISOLATED_SANDBOX_TYPES = frozenset(
    {
        "langsmith",
        "daytona",
        "runloop",
        "e2b",
        "modal",
    }
)


class UnsafeExecutionEnvironmentError(RuntimeError):
    """Raised when autonomous execution would run without isolation."""


def assert_isolated_execution_environment(
    *,
    sandbox_type: str | None = None,
) -> None:
    """Require a known isolated sandbox provider for autonomous execution."""
    raw_value = sandbox_type
    if raw_value is None:
        raw_value = os.environ.get("SANDBOX_TYPE", "langsmith")

    if not isinstance(raw_value, str) or not raw_value.strip():
        raise UnsafeExecutionEnvironmentError(
            "Autonomous execution requires a configured isolated sandbox"
        )

    resolved = raw_value.strip().lower()

    if resolved == "local":
        raise UnsafeExecutionEnvironmentError(
            "Autonomous execution is blocked for SANDBOX_TYPE=local"
        )

    if resolved not in ISOLATED_SANDBOX_TYPES:
        raise UnsafeExecutionEnvironmentError(
            f"Unsupported autonomous execution sandbox type: {resolved}"
        )
