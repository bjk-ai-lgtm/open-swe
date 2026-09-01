import pytest

from agent.orchestration.validation_profiles import (
    ValidationProfile,
    validation_checks_for_profile,
)


def test_none_profile_has_no_checks() -> None:
    assert validation_checks_for_profile(ValidationProfile.NONE) == ()


def test_open_swe_python_profile_has_checks() -> None:
    checks = validation_checks_for_profile(ValidationProfile.OPEN_SWE_PYTHON)

    assert checks
    assert all(check.command for check in checks)


def test_unknown_validation_profile_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported validation profile",
    ):
        validation_checks_for_profile("unknown-profile")


def test_auto_profile_requires_runtime_context() -> None:
    with pytest.raises(
        ValueError,
        match="requires repository runtime context",
    ):
        validation_checks_for_profile(ValidationProfile.AUTO)
