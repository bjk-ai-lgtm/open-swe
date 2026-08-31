import pytest

from agent.dashboard.options import SUPPORTED_MODEL_IDS
from agent.routing import (
    MODEL_ROUTES,
    ORCHESTRATOR_MODEL_ROUTE,
    SpecialistRole,
    has_next_escalation,
    model_for_role,
)


def test_backend_model_escalation_chain() -> None:
    assert model_for_role(SpecialistRole.BACKEND) == "openai:gpt-5.6-terra"
    assert model_for_role(SpecialistRole.BACKEND, escalation_level=1) == "openai:gpt-5.6-sol"
    assert model_for_role(SpecialistRole.BACKEND, escalation_level=2) == "anthropic:claude-opus-5"


def test_backend_has_two_escalation_levels() -> None:
    assert has_next_escalation(SpecialistRole.BACKEND, 0) is True
    assert has_next_escalation(SpecialistRole.BACKEND, 1) is True
    assert has_next_escalation(SpecialistRole.BACKEND, 2) is False


def test_requesting_unknown_escalation_level_fails() -> None:
    with pytest.raises(ValueError):
        model_for_role(
            SpecialistRole.BACKEND,
            escalation_level=3,
        )


def test_all_configured_models_are_supported_by_open_swe() -> None:
    configured = {
        ORCHESTRATOR_MODEL_ROUTE.primary,
        *ORCHESTRATOR_MODEL_ROUTE.escalation,
    }

    for route in MODEL_ROUTES.values():
        configured.add(route.primary)
        configured.update(route.escalation)

    assert configured <= SUPPORTED_MODEL_IDS


def test_role_defaults() -> None:
    assert model_for_role(SpecialistRole.RESEARCH) == "google_genai:gemini-3.7-flash"
    assert model_for_role(SpecialistRole.QA) == "openai:gpt-5.6-terra"
    assert model_for_role(SpecialistRole.GENERAL) == "openai:gpt-5.6-sol"
