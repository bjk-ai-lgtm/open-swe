import pytest

from agent.orchestration.model_factory import (
    build_orchestration_model_factory,
)


def test_openai_model_uses_provider_configuration() -> None:
    calls = []

    def builder(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return object()

    factory = build_orchestration_model_factory(
        use_gateway=True,
        model_effort="high",
        max_tokens=12345,
        model_builder=builder,
    )

    factory("openai:gpt-5.6-terra")

    assert len(calls) == 1

    model_id, kwargs = calls[0]

    assert model_id == "openai:gpt-5.6-terra"
    assert kwargs["use_gateway"] is True
    assert kwargs["max_tokens"] == 12345
    assert kwargs["reasoning"]["effort"] == "high"


def test_anthropic_model_uses_provider_configuration() -> None:
    calls = []

    def builder(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return object()

    factory = build_orchestration_model_factory(
        model_effort="high",
        max_tokens=54321,
        model_builder=builder,
    )

    factory("anthropic:claude-opus-5")

    model_id, kwargs = calls[0]

    assert model_id == "anthropic:claude-opus-5"
    assert kwargs["max_tokens"] == 54321
    assert kwargs["effort"] == "high"
    assert kwargs["thinking"]["type"] == "adaptive"


def test_model_factory_rejects_invalid_token_budget() -> None:
    with pytest.raises(
        ValueError,
        match="max_tokens must be positive",
    ):
        build_orchestration_model_factory(
            max_tokens=0,
        )
