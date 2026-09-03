"""Phase-oriented LangGraph for durable task orchestration."""

from collections.abc import Mapping, Sequence
from typing import NotRequired, Protocol, cast

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from agent.routing import (
    SpecialistRole,
    build_execution_plan,
    model_for_role,
)
from agent.validation import (
    ValidationCheck,
    ValidationReport,
)

from .coordinator import SpecialistExecutionResult
from .durable_state import (
    DurableExecutionSnapshot,
    DurableValidationCheck,
    restore_execution_state,
    restore_validation_checks,
    snapshot_execution_state,
    snapshot_validation_checks,
)
from .server_bridge import PreparedRun
from .state import (
    ExecutionState,
    TaskStatus,
    begin_attempt,
    begin_escalated_attempt,
    create_execution_state,
    mark_execution_complete,
    mark_publication_complete,
    quarantine_task,
    record_execution_failure,
    record_publication_failure,
    record_validation_result,
)


class DurablePhaseService(Protocol):
    """Runtime operations invoked by durable graph phases."""

    async def prepare_run(
        self,
        work_dir: str,
    ) -> PreparedRun: ...

    async def execute_attempt(
        self,
        *,
        task: str,
        work_dir: str,
        role: SpecialistRole,
        model_id: str,
        attempt: int,
        escalation_level: int,
        previous_failure: str | None,
    ) -> SpecialistExecutionResult: ...

    async def validate_attempt(
        self,
        *,
        work_dir: str,
        checks: Sequence[ValidationCheck],
    ) -> ValidationReport: ...

    async def publish_task(
        self,
        *,
        task: str,
        work_dir: str,
    ) -> None: ...


class DurableOrchestratorState(MessagesState):
    """Checkpointed state for phase-oriented orchestration."""

    durable_execution: NotRequired[
        DurableExecutionSnapshot
    ]

    durable_validation_checks: NotRequired[
        list[DurableValidationCheck]
    ]

    orchestration_status: NotRequired[str]
    orchestration_attempts: NotRequired[int]
    orchestration_escalation_level: NotRequired[int]
    orchestration_last_failure: NotRequired[
        str | None
    ]
    orchestration_mode: NotRequired[str]
    orchestration_role: NotRequired[
        str | None
    ]
    orchestration_model_id: NotRequired[
        str | None
    ]
    orchestration_validation_required: NotRequired[
        bool
    ]
    orchestration_phase: NotRequired[str]
    orchestration_summary: NotRequired[
        str | None
    ]


def _latest_human_task(
    state: DurableOrchestratorState,
) -> str | None:
    for message in reversed(
        state.get("messages", [])
    ):
        if not isinstance(
            message,
            HumanMessage,
        ):
            continue

        text = (
            message.text.strip()
            if message.text
            else ""
        )

        if text:
            return text

    return None


def _require_execution_state(
    graph_state: DurableOrchestratorState,
) -> tuple[ExecutionState, str]:
    snapshot = graph_state.get(
        "durable_execution"
    )

    if snapshot is None:
        raise RuntimeError(
            "Durable execution snapshot "
            "is unavailable"
        )

    return restore_execution_state(
        cast(
            Mapping[str, object],
            snapshot,
        )
    )


def _restore_checks(
    graph_state: DurableOrchestratorState,
) -> tuple[ValidationCheck, ...]:
    snapshots = graph_state.get(
        "durable_validation_checks",
        [],
    )

    return restore_validation_checks(
        cast(
            Sequence[
                Mapping[str, object]
            ],
            snapshots,
        )
    )


def _execution_update(
    state: ExecutionState,
    *,
    work_dir: str,
    phase: str,
    model_id: str | None = None,
) -> dict[str, object]:
    update: dict[str, object] = {
        "durable_execution":
            snapshot_execution_state(
                state,
                work_dir=work_dir,
            ),
        "orchestration_status":
            state.status.value,
        "orchestration_attempts":
            state.attempt,
        "orchestration_escalation_level":
            state.escalation_level,
        "orchestration_last_failure":
            state.last_failure,
        "orchestration_mode":
            "execute",
        "orchestration_role":
            state.plan.primary_role.value,
        "orchestration_validation_required":
            state.plan.validation.required,
        "orchestration_phase":
            phase,
    }

    if model_id is not None:
        update[
            "orchestration_model_id"
        ] = model_id

    return update


def _route_from_status(
    graph_state: DurableOrchestratorState,
) -> str:
    state, _ = _require_execution_state(
        graph_state
    )

    if state.status in {
        TaskStatus.PENDING,
        TaskStatus.RETRY_REQUIRED,
        TaskStatus.ESCALATION_REQUIRED,
    }:
        return "begin_attempt"

    if state.status is TaskStatus.VALIDATING:
        return "validate"

    if state.status is TaskStatus.PUBLISHING:
        return "publish"

    if state.status in {
        TaskStatus.SUCCEEDED,
        TaskStatus.QUARANTINED,
    }:
        return "finalize"

    raise RuntimeError(
        "No durable route for task "
        f"status {state.status}"
    )


def build_durable_orchestrator_graph(
    *,
    service: DurablePhaseService,
    work_dir: str,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Build orchestration with checkpoints between runtime phases."""
    if not work_dir.strip():
        raise ValueError(
            "Work directory cannot be empty"
        )

    async def prepare(
        graph_state: DurableOrchestratorState,
    ) -> dict[str, object]:
        task = _latest_human_task(
            graph_state
        )
        if task is None:
            return {
                "orchestration_status":
                    "invalid-input",
                "orchestration_attempts":
                    0,
                "orchestration_escalation_level":
                    0,
                "orchestration_last_failure":
                    (
                        "No non-empty human "
                        "task was found"
                    ),
                "orchestration_mode":
                    "execute",
                "orchestration_role":
                    None,
                "orchestration_model_id":
                    None,
                "orchestration_validation_required":
                    False,
                "orchestration_phase":
                    "invalid-input",
            }

        plan = build_execution_plan(
            task
        )

        prepared = await service.prepare_run(
            work_dir
        )

        state = create_execution_state(
            plan
        )

        update = _execution_update(
            state,
            work_dir=prepared.work_dir,
            phase="prepared",
        )

        update[
            "durable_validation_checks"
        ] = snapshot_validation_checks(
            prepared.checks
        )

        return update

    def route_after_prepare(
        graph_state: DurableOrchestratorState,
    ) -> str:
        if (
            graph_state.get(
                "durable_execution"
            )
            is None
        ):
            return "finalize"

        return "begin_attempt"

    async def start_attempt(
        graph_state: DurableOrchestratorState,
    ) -> dict[str, object]:
        state, prepared_work_dir = (
            _require_execution_state(
                graph_state
            )
        )

        if state.status in {
            TaskStatus.PENDING,
            TaskStatus.RETRY_REQUIRED,
        }:
            state = begin_attempt(
                state
            )

        elif (
            state.status
            is TaskStatus.ESCALATION_REQUIRED
        ):
            state = (
                begin_escalated_attempt(
                    state
                )
            )
        else:
            raise RuntimeError(
                "Cannot start durable "
                "attempt from status "
                f"{state.status}"
            )

        model_id = model_for_role(
            state.plan.primary_role,
            escalation_level=(
                state.escalation_level
            ),
        )

        return _execution_update(
            state,
            work_dir=prepared_work_dir,
            phase="attempt-started",
            model_id=model_id,
        )
    async def execute(
        graph_state: DurableOrchestratorState,
    ) -> dict[str, object]:
        state, prepared_work_dir = (
            _require_execution_state(
                graph_state
            )
        )

        if state.status is not (
            TaskStatus.RUNNING
        ):
            raise RuntimeError(
                "Durable execution phase "
                "requires RUNNING status"
            )

        model_id = model_for_role(
            state.plan.primary_role,
            escalation_level=(
                state.escalation_level
            ),
        )

        try:
            execution = (
                await service.execute_attempt(
                    task=state.plan.task,
                    work_dir=prepared_work_dir,
                    role=(
                        state.plan.primary_role
                    ),
                    model_id=model_id,
                    attempt=state.attempt,
                    escalation_level=(
                        state.escalation_level
                    ),
                    previous_failure=(
                        state.last_failure
                    ),
                )
            )

        except Exception as exc:
            execution = (
                SpecialistExecutionResult(
                    success=False,
                    failure_reason=(
                        "Specialist executor "
                        "raised "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )

        if execution.success:
            state = mark_execution_complete(
                state
            )
        else:
            state = record_execution_failure(
                state,
                failure_reason=(
                    execution.failure_reason
                ),
            )

        update = _execution_update(
            state,
            work_dir=prepared_work_dir,
            phase="execution-complete",
            model_id=model_id,
        )

        if execution.success:
            update[
                "orchestration_summary"
            ] = execution.summary

        return update
    async def validate(
        graph_state: DurableOrchestratorState,
    ) -> dict[str, object]:
        state, prepared_work_dir = (
            _require_execution_state(
                graph_state
            )
        )

        if state.status is not (
            TaskStatus.VALIDATING
        ):
            raise RuntimeError(
                "Durable validation phase "
                "requires VALIDATING status"
            )

        checks = _restore_checks(
            graph_state
        )

        try:
            report = (
                await service.validate_attempt(
                    work_dir=(
                        prepared_work_dir
                    ),
                    checks=checks,
                )
            )
            state = (
                record_validation_result(
                    state,
                    passed=report.passed,
                    failure_reason=(
                        None
                        if report.passed
                        else (
                            report
                            .failure_summary()
                        )
                    ),
                )
            )

        except Exception as exc:
            state = quarantine_task(
                state,
                reason=(
                    "Validation "
                    "infrastructure failure: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )
        return _execution_update(
            state,
            work_dir=prepared_work_dir,
            phase="validation-complete",
            model_id=graph_state.get(
                "orchestration_model_id"
            ),
        )

    async def publish(
        graph_state: DurableOrchestratorState,
    ) -> dict[str, object]:
        state, prepared_work_dir = (
            _require_execution_state(
                graph_state
            )
        )

        if state.status is not (
            TaskStatus.PUBLISHING
        ):
            raise RuntimeError(
                "Durable publication phase "
                "requires PUBLISHING status"
            )
        try:
            await service.publish_task(
                task=state.plan.task,
                work_dir=prepared_work_dir,
            )

        except Exception as exc:
            state = (
                record_publication_failure(
                    state,
                    failure_reason=(
                        "Publication failed: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )

        else:
            state = (
                mark_publication_complete(
                    state
                )
            )
        return _execution_update(
            state,
            work_dir=prepared_work_dir,
            phase="publication-complete",
            model_id=graph_state.get(
                "orchestration_model_id"
            ),
        )

    async def finalize(
        graph_state: DurableOrchestratorState,
    ) -> dict[str, object]:
        snapshot = graph_state.get(
            "durable_execution"
        )

        if snapshot is None:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "No executable task "
                            "was provided."
                        )
                    )
                ],
                "orchestration_phase":
                    "finished",
            }
        state, prepared_work_dir = (
            _require_execution_state(
                graph_state
            )
        )

        if state.status is (
            TaskStatus.SUCCEEDED
        ):
            summary = (
                graph_state.get(
                    "orchestration_summary"
                )
                or (
                    "Task completed "
                    "successfully."
                )
            )

            content = summary

        elif state.status is (
            TaskStatus.QUARANTINED
        ):
            reason = (
                state.last_failure
                or "Unknown failure"
            )

            content = (
                "Task quarantined after "
                f"{state.attempt} "
                "attempt(s). "
                "Last failure: "
                f"{reason}"
            )
        else:
            raise RuntimeError(
                "Cannot finalize "
                "non-terminal durable "
                f"status {state.status}"
            )

        update = _execution_update(
            state,
            work_dir=prepared_work_dir,
            phase="finished",
            model_id=graph_state.get(
                "orchestration_model_id"
            ),
        )

        update["messages"] = [
            AIMessage(
                content=content
            )
        ]

        return update

    async def prepare_node(
        state: DurableOrchestratorState,
    ) -> dict[str, object]:
        return await prepare(state)

    async def start_attempt_node(
        state: DurableOrchestratorState,
    ) -> dict[str, object]:
        return await start_attempt(state)

    async def execute_node(
        state: DurableOrchestratorState,
    ) -> dict[str, object]:
        return await execute(state)

    async def validate_node(
        state: DurableOrchestratorState,
    ) -> dict[str, object]:
        return await validate(state)

    async def publish_node(
        state: DurableOrchestratorState,
    ) -> dict[str, object]:
        return await publish(state)

    async def finalize_node(
        state: DurableOrchestratorState,
    ) -> dict[str, object]:
        return await finalize(state)

    builder = StateGraph(
        DurableOrchestratorState
    )

    builder.add_node(
        "prepare",
        prepare_node,
    )
    builder.add_node(
        "begin_attempt",
        start_attempt_node,
    )

    builder.add_node(
        "execute",
        execute_node,
    )

    builder.add_node(
        "validate",
        validate_node,
    )

    builder.add_node(
        "publish",
        publish_node,
    )

    builder.add_node(
        "finalize",
        finalize_node,
    )

    builder.add_edge(
        START,
        "prepare",
    )

    builder.add_conditional_edges(
        "prepare",
        route_after_prepare,
        {
            "begin_attempt":
                "begin_attempt",
            "finalize":
                "finalize",
        },
    )

    builder.add_edge(
        "begin_attempt",
        "execute",
    )

    builder.add_conditional_edges(
        "execute",
        _route_from_status,
        {
            "begin_attempt":
                "begin_attempt",
            "validate":
                "validate",
            "publish":
                "publish",
            "finalize":
                "finalize",
        },
    )
    builder.add_conditional_edges(
        "validate",
        _route_from_status,
        {
            "begin_attempt":
                "begin_attempt",
            "publish":
                "publish",
            "finalize":
                "finalize",
        },
    )

    builder.add_edge(
        "publish",
        "finalize",
    )

    builder.add_edge(
        "finalize",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer,
    )
