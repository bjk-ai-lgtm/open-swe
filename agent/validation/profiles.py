"""Reusable deterministic validation profiles."""

from .types import ValidationCheck

OPEN_SWE_PYTHON_CHECKS: tuple[ValidationCheck, ...] = (
    ValidationCheck(
        name="tests",
        command=("uv", "run", "pytest", "-q"),
    ),
    ValidationCheck(
        name="ruff",
        command=("uv", "run", "ruff", "check", "."),
    ),
)
