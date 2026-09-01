"""Repository-aware deterministic validation planning."""

import json
import re
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


def _normalized_requirement_name(requirement: object) -> str | None:
    if not isinstance(requirement, str):
        return None

    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if match is None:
        return None

    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def _python_uv_extra(config: Mapping[str, object], package_name: str) -> str | None:
    project = config.get("project")
    if not isinstance(project, dict):
        return None

    optional_dependencies = project.get("optional-dependencies")
    if not isinstance(optional_dependencies, dict):
        return None

    normalized_package = package_name.lower()

    matching_extras = []
    for extra, requirements in optional_dependencies.items():
        if not isinstance(extra, str) or not isinstance(requirements, list):
            continue

        if any(
            _normalized_requirement_name(requirement) == normalized_package
            for requirement in requirements
        ):
            matching_extras.append(extra)

    if not matching_extras:
        return None

    if "dev" in matching_extras:
        return "dev"

    return sorted(matching_extras)[0]


def _python_uv_command(
    config: Mapping[str, object],
    package_name: str,
    *args: str,
) -> tuple[str, ...]:
    extra = _python_uv_extra(config, package_name)

    if extra is not None:
        return ("uv", "run", "--extra", extra, package_name, *args)

    return ("uv", "run", package_name, *args)


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
            _python_uv_command(config, "pytest", "-q")
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
            _python_uv_command(config, "ruff", "check", ".")
            if "uv.lock" in files
            else ("ruff", "check", ".")
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
