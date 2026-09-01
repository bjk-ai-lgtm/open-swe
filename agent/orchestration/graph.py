"""Deterministic LangGraph entry point for custom orchestration."""

from typing import NotRequired, Protocol

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from agent.routing import build_execution_plan, model_for_role

from .coordinator import CoordinatorResult
from .state import TaskStatus


class OrchestrationService(Protocol):
    """Runtime service consumed by the orchestration graph."""

    async def run(
        self,
        *,
        task: str,
        work_dir: str,
    ) -> CoordinatorResult: ...


class OrchestratorGraphState(MessagesState):
    """Observable state emitted by the orchestration graph."""

    orchestration_status: NotRequired[str]
    orchestration_attempts: NotRequired[int]
    orchestration_escalation_level: NotRequired[int]
    orchestration_last_failure: NotRequired[str | None]
    orchestration_mode: NotRequired[str]
    orchestration_role: NotRequired[str | None]
    orchestration_model_id: NotRequired[str | None]
    orchestration_validation_required: NotRequired[bool]


def _latest_human_task(
    state: OrchestratorGraphState,
) -> str | None:
    messages = state.get("messages", [])

    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue

        text = message.text.strip() if message.text else ""

        if text:
            return text

    return None


def _render_result(result: CoordinatorResult) -> str:
    if result.state.status is TaskStatus.SUCCEEDED:
        for attempt in reversed(result.attempts):
            summary = attempt.execution.summary.strip()

            if summary:
                return summary

        return "Task completed successfully."

    reason = result.state.last_failure or "Unknown failure"

    return f"Task quarantined after {result.state.attempt} attempt(s). Last failure: {reason}"


def build_orchestrator_graph(
    *,
    service: OrchestrationService,
    work_dir: str,
    dry_run: bool = False,
):
    """Build the deterministic orchestration graph."""
    if not work_dir.strip():
        raise ValueError("Work directory cannot be empty")

    async def orchestrate(
        state: OrchestratorGraphState,
    ) -> dict:
        task = _latest_human_task(state)

        if task is None:
            return {
                "messages": [AIMessage(content="No executable task was provided.")],
                "orchestration_status": "invalid-input",
                "orchestration_attempts": 0,
                "orchestration_escalation_level": 0,
                "orchestration_last_failure": ("No non-empty human task was found"),
                "orchestration_mode": ("dry-run" if dry_run else "execute"),
                "orchestration_role": None,
                "orchestration_model_id": None,
                "orchestration_validation_required": False,
            }

        if dry_run:
            plan = build_execution_plan(task)

            model_id = model_for_role(
                plan.primary_role,
                escalation_level=0,
            )

            validation_required = plan.validation.required

            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Dry-run plan created. "
                            f"Role: {plan.primary_role.value}. "
                            f"Model: {model_id}. "
                            "Validation required: "
                            f"{validation_required}."
                        )
                    )
                ],
                "orchestration_status": "planned",
                "orchestration_attempts": 0,
                "orchestration_escalation_level": 0,
                "orchestration_last_failure": None,
                "orchestration_mode": "dry-run",
                "orchestration_role": (plan.primary_role.value),
                "orchestration_model_id": model_id,
                "orchestration_validation_required": (validation_required),
            }

        result = await service.run(
            task=task,
            work_dir=work_dir,
        )

        last_attempt = result.attempts[-1] if result.attempts else None

        return {
            "messages": [
                AIMessage(
                    content=_render_result(result),
                )
            ],
            "orchestration_status": (result.state.status.value),
            "orchestration_attempts": (result.state.attempt),
            "orchestration_escalation_level": (result.state.escalation_level),
            "orchestration_last_failure": (result.state.last_failure),
            "orchestration_mode": "execute",
            "orchestration_role": (result.state.plan.primary_role.value),
            "orchestration_model_id": (last_attempt.model_id if last_attempt else None),
            "orchestration_validation_required": (result.state.plan.validation.required),
        }

    builder = StateGraph(OrchestratorGraphState)

    builder.add_node(
        "orchestrate",
        orchestrate,
    )

    builder.add_edge(
        START,
        "orchestrate",
    )

    builder.add_edge(
        "orchestrate",
        END,
    )

    return builder.compile()
