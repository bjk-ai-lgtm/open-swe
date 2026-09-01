"""Deterministic LangGraph entry point for custom orchestration."""

from typing import NotRequired, Protocol

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

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
            }

        result = await service.run(
            task=task,
            work_dir=work_dir,
        )

        return {
            "messages": [
                AIMessage(
                    content=_render_result(result),
                )
            ],
            "orchestration_status": result.state.status.value,
            "orchestration_attempts": result.state.attempt,
            "orchestration_escalation_level": (result.state.escalation_level),
            "orchestration_last_failure": (result.state.last_failure),
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
