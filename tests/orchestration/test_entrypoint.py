import json
from pathlib import Path

from agent.graphs.orchestrator import (
    get_orchestrator,
    traced_orchestrator,
)


def test_orchestrator_entrypoint_is_importable() -> None:
    assert callable(get_orchestrator)
    assert callable(traced_orchestrator)


def test_langgraph_config_registers_orchestrator() -> None:
    root = Path(__file__).resolve().parents[2]

    config = json.loads((root / "langgraph.json").read_text())

    assert config["graphs"]["orchestrator"] == ("agent.graphs.orchestrator:traced_orchestrator")
