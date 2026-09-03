"""Serializable durable state for orchestration checkpoints."""

from collections.abc import Mapping, Sequence
from typing import TypedDict

from agent.routing import build_execution_plan
from agent.validation import ValidationCheck

from .state import ExecutionState, TaskStatus

DURABLE_STATE_SCHEMA_VERSION = 1


class DurableExecutionSnapshot(TypedDict):
    """Primitive-only representation suitable for durable checkpoints."""

    schema_version: int
    task: str
    work_dir: str
    status: str
    attempt: int
    execution_failures: int
    validation_failures: int
    escalation_level: int
    last_failure: str | None


def snapshot_execution_state(
    state: ExecutionState,
    *,
    work_dir: str,
) -> DurableExecutionSnapshot:
    """Convert runtime execution state into durable primitive values."""
    normalized_work_dir = work_dir.strip()

    if not normalized_work_dir:
        raise ValueError(
            "Durable execution work directory cannot be empty"
        )

    return {
        "schema_version": DURABLE_STATE_SCHEMA_VERSION,
        "task": state.plan.task,
        "work_dir": normalized_work_dir,
        "status": state.status.value,
        "attempt": state.attempt,
        "execution_failures": state.execution_failures,
        "validation_failures": state.validation_failures,
        "escalation_level": state.escalation_level,
        "last_failure": state.last_failure,
    }


def _required_string(
    snapshot: Mapping[str, object],
    key: str,
) -> str:
    value = snapshot.get(key)

    if not isinstance(value, str):
        raise ValueError(
            f"Durable execution field {key!r} "
            "must be a non-empty string"
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"Durable execution field {key!r} "
            "must be a non-empty string"
        )

    return normalized


def _non_negative_int(
    snapshot: Mapping[str, object],
    key: str,
) -> int:
    value = snapshot.get(key)

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ValueError(
            f"Durable execution field {key!r} "
            "must be a non-negative integer"
        )

    return value


def restore_execution_state(
    snapshot: Mapping[str, object],
) -> tuple[ExecutionState, str]:
    """Restore deterministic runtime state from a durable snapshot."""
    schema_version = snapshot.get(
        "schema_version"
    )

    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version
        != DURABLE_STATE_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported durable execution "
            f"schema version: {schema_version!r}"
        )

    task: str = _required_string(
        snapshot,
        "task",
    )

    work_dir: str = _required_string(
        snapshot,
        "work_dir",
    )

    raw_status: str = _required_string(
        snapshot,
        "status",
    )

    try:
        status = TaskStatus(
            raw_status
        )
    except ValueError as exc:
        raise ValueError(
            "Unsupported durable execution "
            f"status: {raw_status!r}"
        ) from exc

    last_failure = snapshot.get(
        "last_failure"
    )

    if (
        last_failure is not None
        and not isinstance(
            last_failure,
            str,
        )
    ):
        raise ValueError(
            "Durable execution field "
            "'last_failure' must be "
            "a string or None"
        )

    plan = build_execution_plan(
        task
    )

    state = ExecutionState(
        plan=plan,
        status=status,
        attempt=_non_negative_int(
            snapshot,
            "attempt",
        ),
        execution_failures=_non_negative_int(
            snapshot,
            "execution_failures",
        ),
        validation_failures=_non_negative_int(
            snapshot,
            "validation_failures",
        ),
        escalation_level=_non_negative_int(
            snapshot,
            "escalation_level",
        ),
        last_failure=last_failure,
    )

    return state, work_dir


class DurableValidationCheck(TypedDict):
    """Primitive representation of one validation check."""

    name: str
    command: list[str]
    required: bool


def snapshot_validation_checks(
    checks: tuple[ValidationCheck, ...],
) -> list[DurableValidationCheck]:
    """Serialize deterministic validation checks for checkpoints."""
    snapshots: list[
        DurableValidationCheck
    ] = []

    for check in checks:
        name = check.name.strip()

        if not name:
            raise ValueError(
                "Validation check name "
                "cannot be empty"
            )

        if not check.command:
            raise ValueError(
                "Validation check command "
                "cannot be empty"
            )

        command = list(
            check.command
        )

        if any(
            not part.strip()
            for part in command
        ):
            raise ValueError(
                "Validation check command "
                "parts cannot be empty"
            )

        snapshots.append(
            {
                "name": name,
                "command": command,
                "required": check.required,
            }
        )

    return snapshots


def restore_validation_checks(
    snapshots: Sequence[
        Mapping[str, object]
    ],
) -> tuple[ValidationCheck, ...]:
    """Restore deterministic checks from checkpoint data."""
    checks: list[ValidationCheck] = []

    for snapshot in snapshots:
        name = snapshot.get(
            "name"
        )

        command = snapshot.get(
            "command"
        )

        required = snapshot.get(
            "required"
        )

        if (
            not isinstance(name, str)
            or not name.strip()
        ):
            raise ValueError(
                "Durable validation check "
                "name must be non-empty"
            )

        if (
            not isinstance(command, list)
            or not command
            or any(
                not isinstance(part, str)
                or not part.strip()
                for part in command
            )
        ):
            raise ValueError(
                "Durable validation check "
                "command must contain "
                "non-empty strings"
            )

        if not isinstance(
            required,
            bool,
        ):
            raise ValueError(
                "Durable validation check "
                "required flag must be boolean"
            )

        checks.append(
            ValidationCheck(
                name=name.strip(),
                command=tuple(
                    part.strip()
                    for part in command
                ),
                required=required,
            )
        )

    return tuple(checks)
