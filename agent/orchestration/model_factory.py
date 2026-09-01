"""Provider-aware model construction for routed specialists."""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from agent.runtime import DEFAULT_LLM_MAX_TOKENS
from agent.utils.model import make_model, provider_model_kwargs

ModelBuilder = Callable[..., BaseChatModel]


def build_orchestration_model_factory(
    *,
    use_gateway: bool | None = None,
    model_effort: str | None = None,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    model_builder: ModelBuilder = make_model,
) -> Callable[[str], BaseChatModel]:
    """Build models using Open SWE's provider-specific configuration."""
    if max_tokens <= 0:
        raise ValueError("Model max_tokens must be positive")

    def factory(model_id: str) -> BaseChatModel:
        kwargs = provider_model_kwargs(
            model_id,
            model_effort,
            max_tokens=max_tokens,
        )

        return model_builder(
            model_id,
            use_gateway=use_gateway,
            **kwargs,
        )

    return factory
