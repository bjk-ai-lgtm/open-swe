from dataclasses import replace

import pytest

from agent.orchestration.durable_state import (
    DURABLE_STATE_SCHEMA_VERSION,
    restore_execution_state,
    restore_validation_checks,
    snapshot_execution_state,
    snapshot_validation_checks,
)
from agent.orchestration.state import (
    ExecutionState,
    TaskStatus,
    create_execution_state,
)
from agent.routing import build_execution_plan
from agent.validation import ValidationCheck

TASK = (
    "Implement a REST API backed "
    "by the database."
)


def make_state() -> ExecutionState:
    plan = build_execution_plan(
        TASK
    )

    initial = create_execution_state(
        plan
    )

    return replace(
        initial,
        status=TaskStatus.PUBLISHING,
        attempt=3,
        execution_failures=1,
        validation_failures=2,
        escalation_level=1,
        last_failure="previous failure",
    )


def test_snapshot_round_trips_execution_state() -> None:
    original = make_state()

    snapshot = snapshot_execution_state(
        original,
        work_dir="/workspace/open-swe",
    )

    restored, work_dir = (
        restore_execution_state(
            snapshot
        )
    )

    assert (
        snapshot["schema_version"]
        == DURABLE_STATE_SCHEMA_VERSION
    )

    assert work_dir == (
        "/workspace/open-swe"
    )

    assert restored.status is (
        TaskStatus.PUBLISHING
    )

    assert restored.attempt == 3
    assert restored.execution_failures == 1
    assert restored.validation_failures == 2
    assert restored.escalation_level == 1

    assert restored.last_failure == (
        "previous failure"
    )

    assert restored.plan == (
        original.plan
    )


def test_snapshot_contains_only_primitive_values() -> None:
    snapshot = snapshot_execution_state(
        make_state(),
        work_dir="/workspace/open-swe",
    )

    for value in snapshot.values():
        assert (
            value is None
            or isinstance(
                value,
                (str, int),
            )
        )


def test_snapshot_rejects_empty_work_dir() -> None:
    with pytest.raises(
        ValueError,
        match="work directory",
    ):
        snapshot_execution_state(
            make_state(),
            work_dir="   ",
        )


def test_restore_rejects_unknown_schema() -> None:
    snapshot: dict[str, object] = dict(
        snapshot_execution_state(
            make_state(),
            work_dir="/workspace/open-swe",
        )
    )

    snapshot["schema_version"] = 999

    with pytest.raises(
        ValueError,
        match="schema version",
    ):
        restore_execution_state(
            snapshot
        )


def test_restore_rejects_unknown_status() -> None:
    snapshot: dict[str, object] = dict(
        snapshot_execution_state(
            make_state(),
            work_dir="/workspace/open-swe",
        )
    )

    snapshot["status"] = "teleporting"

    with pytest.raises(
        ValueError,
        match="status",
    ):
        restore_execution_state(
            snapshot
        )


@pytest.mark.parametrize(
    "field",
    [
        "attempt",
        "execution_failures",
        "validation_failures",
        "escalation_level",
    ],
)
def test_restore_rejects_negative_counters(
    field: str,
) -> None:
    snapshot: dict[str, object] = dict(
        snapshot_execution_state(
            make_state(),
            work_dir="/workspace/open-swe",
        )
    )

    snapshot[field] = -1

    with pytest.raises(
        ValueError,
        match=field,
    ):
        restore_execution_state(
            snapshot
        )


def test_restore_rejects_boolean_counter() -> None:
    snapshot: dict[str, object] = dict(
        snapshot_execution_state(
            make_state(),
            work_dir="/workspace/open-swe",
        )
    )

    snapshot["attempt"] = True

    with pytest.raises(
        ValueError,
        match="attempt",
    ):
        restore_execution_state(
            snapshot
        )


def test_validation_checks_round_trip() -> None:
    checks = (
        ValidationCheck(
            name="pytest",
            command=(
                "pytest",
                "-q",
            ),
            required=True,
        ),
        ValidationCheck(
            name="ruff",
            command=(
                "ruff",
                "check",
                ".",
            ),
            required=False,
        ),
    )

    snapshots = (
        snapshot_validation_checks(
            checks
        )
    )

    assert snapshots == [
        {
            "name": "pytest",
            "command": [
                "pytest",
                "-q",
            ],
            "required": True,
        },
        {
            "name": "ruff",
            "command": [
                "ruff",
                "check",
                ".",
            ],
            "required": False,
        },
    ]

    restored = (
        restore_validation_checks(
            snapshots
        )
    )

    assert restored == checks


def test_validation_check_snapshot_rejects_empty_command() -> None:
    checks = (
        ValidationCheck(
            name="broken",
            command=(),
        ),
    )

    with pytest.raises(
        ValueError,
        match="command",
    ):
        snapshot_validation_checks(
            checks
        )


def test_validation_check_restore_rejects_invalid_required_flag() -> None:
    snapshots = [
        {
            "name": "pytest",
            "command": [
                "pytest",
                "-q",
            ],
            "required": "yes",
        }
    ]

    with pytest.raises(
        ValueError,
        match="required",
    ):
        restore_validation_checks(
            snapshots
        )
