"""Deterministic capability-based model routing."""

from dataclasses import dataclass

from .types import SpecialistRole


@dataclass(frozen=True)
class ModelRoute:
    """Ordered model capability route for one role."""

    primary: str
    escalation: tuple[str, ...] = ()

    @property
    def max_escalation_level(self) -> int:
        return len(self.escalation)

    def model_for_level(self, escalation_level: int) -> str:
        if escalation_level < 0:
            raise ValueError("Escalation level cannot be negative")

        if escalation_level == 0:
            return self.primary

        index = escalation_level - 1

        if index >= len(self.escalation):
            raise ValueError(f"No model configured for escalation level {escalation_level}")

        return self.escalation[index]


ORCHESTRATOR_MODEL_ROUTE = ModelRoute(
    primary="openai:gpt-5.6-sol",
    escalation=("anthropic:claude-opus-5",),
)


MODEL_ROUTES: dict[SpecialistRole, ModelRoute] = {
    SpecialistRole.BACKEND: ModelRoute(
        primary="openai:gpt-5.6-terra",
        escalation=(
            "openai:gpt-5.6-sol",
            "anthropic:claude-opus-5",
        ),
    ),
    SpecialistRole.RESEARCH: ModelRoute(
        primary="google_genai:gemini-3.7-flash",
        escalation=(
            "openai:gpt-5.6-sol",
            "anthropic:claude-opus-5",
        ),
    ),
    SpecialistRole.QA: ModelRoute(
        primary="openai:gpt-5.6-terra",
        escalation=("openai:gpt-5.6-sol",),
    ),
    SpecialistRole.GENERAL: ModelRoute(
        primary="openai:gpt-5.6-sol",
        escalation=("anthropic:claude-opus-5",),
    ),
}


def model_route_for(role: SpecialistRole) -> ModelRoute:
    """Return the configured model route for a specialist role."""
    return MODEL_ROUTES[role]


def model_for_role(
    role: SpecialistRole,
    *,
    escalation_level: int = 0,
) -> str:
    """Select the model for a role and capability escalation level."""
    return model_route_for(role).model_for_level(escalation_level)


def has_next_escalation(
    role: SpecialistRole,
    escalation_level: int,
) -> bool:
    """Return whether a stronger capability tier remains."""
    route = model_route_for(role)
    return escalation_level < route.max_escalation_level
