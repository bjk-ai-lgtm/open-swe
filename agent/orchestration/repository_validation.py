"""Repository-aware deterministic validation planning."""

import json
import tomllib
from collections.abc import Mapping
from enum import StrEnum

from agent.validation import ValidationCheck


class RepositoryFamily(StrEnum):
    """Supported repository families for automatic validation."""

    PYTHON = "python"
    NODE = "node"
    GO = "go"
    UNKNOWN = "unknown"


def detect_repository_family(files: Mapping[str, str]) -> RepositoryFamily:
    """Detect a repository family from deterministic project markers."""
    names = set(files)

    if "pyproject.toml" in names:
        return RepositoryFamily.PYTHON

    if "package.json" in names:
        return RepositoryFamily.NODE

    if "go.mod" in names:
        return RepositoryFamily.GO

    return RepositoryFamily.UNKNOWN


def _python_checks(files: Mapping[str, str]) -> tuple[ValidationCheck, ...]:
    raw = files.get("pyproject.toml")
    if raw is None:
        return ()

    try:
        config = tomllib.loads(raw)
    except tomllib.TOMLDecodeError:
        return ()

    tool = config.get("tool")
    tool = tool if isinstance(tool, dict) else {}

    checks: list[ValidationCheck] = []

    if "pytest" in tool:
        command = (
            ("uv", "run", "pytest", "-q")
            if "uv.lock" in files
            else ("python", "-m", "pytest", "-q")
        )
        checks.append(
            ValidationCheck(
                name="pytest",
                command=command,
            )
        )

    if "ruff" in tool:
        command = (
            ("uv", "run", "ruff", "check", ".") if "uv.lock" in files else ("ruff", "check", ".")
        )
        checks.append(
            ValidationCheck(
                name="ruff",
                command=command,
            )
        )

    return tuple(checks)


def _node_checks(files: Mapping[str, str]) -> tuple[ValidationCheck, ...]:
    raw = files.get("package.json")
    if raw is None:
        return ()

    try:
        package = json.loads(raw)
    except json.JSONDecodeError:
        return ()

    scripts = package.get("scripts")
    if not isinstance(scripts, dict):
        return ()

    if "pnpm-lock.yaml" in files:
        runner = ("pnpm",)
    elif "yarn.lock" in files:
        runner = ("yarn",)
    else:
        runner = ("npm", "run")

    checks: list[ValidationCheck] = []

    if isinstance(scripts.get("test"), str):
        checks.append(
            ValidationCheck(
                name="test",
                command=(*runner, "test"),
            )
        )

    if isinstance(scripts.get("lint"), str):
        checks.append(
            ValidationCheck(
                name="lint",
                command=(*runner, "lint"),
            )
        )

    return tuple(checks)


def _go_checks(files: Mapping[str, str]) -> tuple[ValidationCheck, ...]:
    if "go.mod" not in files:
        return ()

    return (
        ValidationCheck(
            name="go-test",
            command=("go", "test", "./..."),
        ),
    )


def validation_checks_for_repository(
    files: Mapping[str, str],
) -> tuple[ValidationCheck, ...]:
    """Resolve deterministic validation checks from repository metadata."""
    family = detect_repository_family(files)

    if family is RepositoryFamily.PYTHON:
        return _python_checks(files)

    if family is RepositoryFamily.NODE:
        return _node_checks(files)

    if family is RepositoryFamily.GO:
        return _go_checks(files)

    return ()
