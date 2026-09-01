import json

from agent.orchestration.repository_validation import (
    RepositoryFamily,
    detect_repository_family,
    validation_checks_for_repository,
)


def test_detects_python_repository():
    files = {
        "pyproject.toml": '[project]\nname = "demo"\n',
    }

    assert detect_repository_family(files) is RepositoryFamily.PYTHON


def test_python_uv_repository_resolves_pytest_and_ruff():
    files = {
        "pyproject.toml": """
[project]
name = "demo"

[tool.pytest.ini_options]

[tool.ruff]
""",
        "uv.lock": "",
    }

    checks = validation_checks_for_repository(files)

    assert [check.name for check in checks] == [
        "pytest",
        "ruff",
    ]
    assert checks[0].command == (
        "uv",
        "run",
        "pytest",
        "-q",
    )
    assert checks[1].command == (
        "uv",
        "run",
        "ruff",
        "check",
        ".",
    )


def test_python_repository_without_declared_tools_is_fail_closed():
    files = {
        "pyproject.toml": """
[project]
name = "demo"
""",
    }

    assert validation_checks_for_repository(files) == ()


def test_node_repository_uses_declared_scripts():
    files = {
        "package.json": json.dumps(
            {
                "scripts": {
                    "test": "vitest run",
                    "lint": "eslint .",
                }
            }
        ),
        "pnpm-lock.yaml": "",
    }

    checks = validation_checks_for_repository(files)

    assert [check.command for check in checks] == [
        ("pnpm", "test"),
        ("pnpm", "lint"),
    ]


def test_node_repository_does_not_invent_missing_scripts():
    files = {
        "package.json": json.dumps(
            {
                "scripts": {
                    "build": "vite build",
                }
            }
        )
    }

    assert validation_checks_for_repository(files) == ()


def test_go_repository_uses_go_test():
    files = {
        "go.mod": "module example.com/demo\n",
    }

    checks = validation_checks_for_repository(files)

    assert len(checks) == 1
    assert checks[0].command == (
        "go",
        "test",
        "./...",
    )


def test_unknown_repository_is_fail_closed():
    files = {
        "README.md": "# Demo",
    }

    assert detect_repository_family(files) is RepositoryFamily.UNKNOWN
    assert validation_checks_for_repository(files) == ()
