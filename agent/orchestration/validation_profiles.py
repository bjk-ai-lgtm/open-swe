"""Deterministic validation profiles for orchestrated repositories."""

from enum import StrEnum

from agent.validation import (
    OPEN_SWE_PYTHON_CHECKS,
    ValidationCheck,
)


class ValidationProfile(StrEnum):
    """Known deterministic validation configurations."""

    NONE = "none"
    OPEN_SWE_PYTHON = "open-swe-python"
    AUTO = "auto"


def validation_checks_for_profile(
    profile: str | ValidationProfile,
) -> tuple[ValidationCheck, ...]:
    """Resolve an explicit validation profile into deterministic checks."""
    try:
        resolved = ValidationProfile(profile)
    except ValueError as exc:
        raise ValueError(f"Unsupported validation profile: {profile}") from exc

    if resolved is ValidationProfile.NONE:
        return ()

    if resolved is ValidationProfile.OPEN_SWE_PYTHON:
        return tuple(OPEN_SWE_PYTHON_CHECKS)

    if resolved is ValidationProfile.AUTO:
        raise ValueError("Automatic validation requires repository runtime context")

    raise AssertionError(f"Unhandled validation profile: {resolved}")
