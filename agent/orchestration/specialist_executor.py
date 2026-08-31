"""Open SWE specialist execution adapter."""

from collections.abc import Callable, Sequence
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT, SubAgent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agent.routing import SpecialistRole
from agent.specialist_agents import build_v01_specialists
from agent.utils.model import make_model

from .coordinator import SpecialistExecutionResult

ModelFactory = Callable[[str], BaseChatModel]
AgentFactory = Callable[..., Any]


class OpenSWESpecialistExecutor:
    """Execute routed specialists using the Open SWE/Deep Agents runtime."""

    def __init__(
        self,
        *,
        backend: BackendProtocol,
        tools: Sequence[Any],
        skills: list[str] | None = None,
        middleware: Sequence[AgentMiddleware[Any, Any, Any]] | None = None,
        use_gateway: bool | None = None,
        model_factory: ModelFactory | None = None,
        agent_factory: AgentFactory = create_deep_agent,
    ) -> None:
        self._backend = backend
        self._tools = list(tools)
        self._skills = list(skills) if skills else None
        self._middleware = list(middleware or [])
        self._use_gateway = use_gateway
        self._agent_factory = agent_factory

        if model_factory is None:

            def default_model_factory(model_id: str) -> BaseChatModel:
                return make_model(
                    model_id,
                    use_gateway=self._use_gateway,
                )

            self._model_factory = default_model_factory
        else:
            self._model_factory = model_factory

    def _specialist_spec(
        self,
        role: SpecialistRole,
        model: BaseChatModel,
    ) -> SubAgent:
        """Build the declarative specialist spec for one routed role."""
        if role is SpecialistRole.GENERAL:
            spec: SubAgent = {
                **GENERAL_PURPOSE_SUBAGENT,
                "model": model,
                "tools": self._tools,
                "middleware": self._middleware,
            }

            if self._skills:
                spec["skills"] = self._skills

            return spec

        specialists = build_v01_specialists(
            model,
            tools=self._tools,
            skills=self._skills,
            middleware=self._middleware,
        )

        for spec in specialists:
            if spec["name"] == role.value:
                return spec

        raise ValueError(f"No specialist implementation registered for role {role}")

    @staticmethod
    def _execution_prompt(
        *,
        task: str,
        work_dir: str,
        attempt: int,
        escalation_level: int,
        previous_failure: str | None,
    ) -> str:
        parts = [
            f"Task: {task}",
            f"Working directory: {work_dir}",
            f"Attempt: {attempt}",
            f"Capability escalation level: {escalation_level}",
        ]

        if previous_failure:
            parts.extend(
                [
                    "",
                    "Previous attempt failure:",
                    previous_failure,
                    "",
                    (
                        "Inspect the previous failure carefully and correct the "
                        "underlying cause rather than repeating the same approach."
                    ),
                ]
            )

        return "\n".join(parts)

    @staticmethod
    def _summary_from_result(result: Any) -> str:
        if not isinstance(result, dict):
            return ""

        messages = result.get("messages")

        if not isinstance(messages, list):
            return ""

        for message in reversed(messages):
            if isinstance(message, AIMessage):
                text = message.text.strip() if message.text else ""
                if text:
                    return text

        return ""

    async def execute(
        self,
        *,
        thread_id: str,
        work_dir: str,
        task: str,
        role: SpecialistRole,
        model_id: str,
        attempt: int,
        escalation_level: int,
        previous_failure: str | None,
    ) -> SpecialistExecutionResult:
        """Execute one coordinator-selected specialist attempt."""
        model = self._model_factory(model_id)

        spec = self._specialist_spec(
            role,
            model,
        )

        agent = self._agent_factory(
            model=model,
            system_prompt=spec["system_prompt"],
            tools=spec.get("tools", []),
            skills=spec.get("skills"),
            backend=self._backend,
            middleware=spec.get("middleware", []),
            subagents=[],
        )

        prompt = self._execution_prompt(
            task=task,
            work_dir=work_dir,
            attempt=attempt,
            escalation_level=escalation_level,
            previous_failure=previous_failure,
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=prompt),
                ]
            },
            {
                "configurable": {
                    "thread_id": thread_id,
                },
                "metadata": {
                    "open_swe_specialist_role": role.value,
                    "open_swe_model_id": model_id,
                    "open_swe_attempt": attempt,
                    "open_swe_escalation_level": escalation_level,
                },
            },
        )

        return SpecialistExecutionResult(
            success=True,
            summary=self._summary_from_result(result),
        )
