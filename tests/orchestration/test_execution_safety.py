import pytest

from agent.orchestration.execution_safety import (
    UnsafeExecutionEnvironmentError,
    assert_isolated_execution_environment,
)


@pytest.mark.parametrize(
    "sandbox_type",
    [
        "langsmith",
        "daytona",
        "runloop",
        "e2b",
        "modal",
    ],
)
def test_known_isolated_sandbox_types_are_allowed(sandbox_type) -> None:
    assert_isolated_execution_environment(
        sandbox_type=sandbox_type,
    )


def test_local_execution_is_blocked() -> None:
    with pytest.raises(
        UnsafeExecutionEnvironmentError,
        match="SANDBOX_TYPE=local",
    ):
        assert_isolated_execution_environment(
            sandbox_type="local",
        )


def test_unknown_execution_environment_is_fail_closed() -> None:
    with pytest.raises(
        UnsafeExecutionEnvironmentError,
        match="Unsupported autonomous execution sandbox type",
    ):
        assert_isolated_execution_environment(
            sandbox_type="mystery-provider",
        )
