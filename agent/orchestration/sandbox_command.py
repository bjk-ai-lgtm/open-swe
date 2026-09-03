"""Provider-aware execution for orchestration control-plane commands."""

import asyncio
import os
import shlex
import uuid
from dataclasses import dataclass
from typing import Any

from deepagents.backends.protocol import SandboxBackendProtocol

from agent.utils.sandbox_state import unwrap_sandbox_backend

DAYTONA_COMMAND_POLL_INTERVAL_SECONDS = 0.2
DAYTONA_PGID_DISCOVERY_TIMEOUT_SECONDS = 5.0
DAYTONA_CONTROL_COMMAND_TIMEOUT_SECONDS = 10
DAYTONA_TERMINATION_GRACE_POLLS = 10


@dataclass(frozen=True)
class SandboxCommandResult:
    """Normalized result for a control-plane sandbox command."""

    exit_code: int
    output: str


def _daytona_process(
    sandbox_backend: SandboxBackendProtocol,
) -> Any:
    backend = unwrap_sandbox_backend(sandbox_backend)

    process = getattr(
        getattr(backend, "_sandbox", None),
        "process",
        None,
    )

    required = (
        "create_session",
        "execute_session_command",
        "get_session_command",
        "get_session_command_logs",
        "delete_session",
        "exec",
    )

    missing = [
        name
        for name in required
        if not callable(getattr(process, name, None))
    ]

    if missing:
        raise RuntimeError(
            "Daytona backend does not expose required "
            "native process APIs: "
            + ", ".join(missing)
        )

    return process


async def _daytona_exec(
    process: Any,
    command: str,
) -> Any:
    return await asyncio.to_thread(
        process.exec,
        command,
        timeout=DAYTONA_CONTROL_COMMAND_TIMEOUT_SECONDS,
    )


async def _discover_pgid(
    process: Any,
    pid_file: str,
) -> int | None:
    loop = asyncio.get_running_loop()
    deadline = (
        loop.time()
        + DAYTONA_PGID_DISCOVERY_TIMEOUT_SECONDS
    )

    rendered = shlex.quote(pid_file)

    while True:
        result = await _daytona_exec(
            process,
            f"cat {rendered} 2>/dev/null || true",
        )

        raw = getattr(result, "result", "")
        value = raw.strip() if isinstance(raw, str) else ""

        if value.isdigit():
            return int(value)

        if loop.time() >= deadline:
            return None

        await asyncio.sleep(0.1)


async def _terminate_process_group(
    process: Any,
    pgid: int,
    pid_file: str,
) -> None:
    rendered = shlex.quote(pid_file)

    command = f"""
set +e

/bin/kill -TERM -- -{pgid} 2>/dev/null || true

i=0
while [ "$i" -lt {DAYTONA_TERMINATION_GRACE_POLLS} ]; do
    if ! /bin/kill -0 -- -{pgid} 2>/dev/null; then
        rm -f {rendered}
        exit 0
    fi

    i=$((i + 1))
    sleep 0.2
done

/bin/kill -KILL -- -{pgid} 2>/dev/null || true

sleep 0.1

if /bin/kill -0 -- -{pgid} 2>/dev/null; then
    exit 70
fi

rm -f {rendered}
""".strip()

    result = await _daytona_exec(
        process,
        command,
    )

    if getattr(result, "exit_code", None) != 0:
        raise RuntimeError(
            "Unable to terminate Daytona "
            "control-plane process group "
            f"{pgid}"
        )


async def _cleanup_session(
    process: Any,
    *,
    session_id: str,
    pid_file: str,
    pgid: int | None,
    terminate: bool,
) -> None:
    errors: list[str] = []

    if terminate:
        try:
            resolved_pgid = pgid

            if resolved_pgid is None:
                resolved_pgid = await _discover_pgid(
                    process,
                    pid_file,
                )

            if resolved_pgid is None:
                raise RuntimeError(
                    "Unable to discover Daytona "
                    "control-plane process group "
                    "for cancellation"
                )

            await _terminate_process_group(
                process,
                resolved_pgid,
                pid_file,
            )

        except Exception as exc:
            errors.append(
                "process termination failed: "
                f"{exc}"
            )

    rendered = shlex.quote(pid_file)

    try:
        cleanup = await _daytona_exec(
            process,
            f"rm -f {rendered}",
        )

        if getattr(
            cleanup,
            "exit_code",
            None,
        ) != 0:
            raise RuntimeError(
                "Unable to remove Daytona "
                "control-plane PID file"
            )

    except Exception as exc:
        errors.append(
            "PID cleanup failed: "
            f"{exc}"
        )

    try:
        await asyncio.to_thread(
            process.delete_session,
            session_id,
        )

    except Exception as exc:
        errors.append(
            "session deletion failed: "
            f"{exc}"
        )

    if errors:
        raise RuntimeError(
            "Daytona control-plane cleanup failed: "
            + "; ".join(errors)
        )


def _wrapped_daytona_command(
    command: str,
    pid_file: str,
) -> str:
    rendered = shlex.quote(pid_file)

    inner = (
        f"printf '%s\\n' \"$$\" > {rendered}; "
        f"{command}"
    )

    return (
        "setsid sh -c "
        f"{shlex.quote(inner)}"
    )


def _clean_logs(logs: Any) -> str:
    stdout = getattr(logs, "stdout", "") or ""
    stderr = getattr(logs, "stderr", "") or ""

    if not isinstance(stdout, str):
        stdout = str(stdout)

    if not isinstance(stderr, str):
        stderr = str(stderr)

    return stdout + stderr


async def _execute_daytona_command(
    sandbox_backend: SandboxBackendProtocol,
    command: str,
    *,
    timeout: int,
) -> SandboxCommandResult:
    from daytona import SessionExecuteRequest

    process = _daytona_process(
        sandbox_backend,
    )

    token = uuid.uuid4().hex

    session_id = (
        f"open-swe-control-{token}"
    )

    pid_file = (
        f"/tmp/open-swe-control-{token}.pgid"
    )

    session_created = False
    launch_attempted = False
    command_completed = False
    pgid: int | None = None

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    try:
        await asyncio.to_thread(
            process.create_session,
            session_id,
        )

        session_created = True
        launch_attempted = True

        response = await asyncio.to_thread(
            process.execute_session_command,
            session_id,
            SessionExecuteRequest(
                command=_wrapped_daytona_command(
                    command,
                    pid_file,
                ),
                run_async=True,
            ),
            DAYTONA_CONTROL_COMMAND_TIMEOUT_SECONDS,
        )

        command_id = getattr(
            response,
            "cmd_id",
            None,
        )

        if (
            not isinstance(command_id, str)
            or not command_id
        ):
            raise RuntimeError(
                "Daytona session command returned "
                "an invalid command ID"
            )

        pgid = await _discover_pgid(
            process,
            pid_file,
        )

        if pgid is None:
            raise RuntimeError(
                "Daytona control-plane command "
                "did not publish its "
                "process-group ID"
            )

        while True:
            remote = await asyncio.to_thread(
                process.get_session_command,
                session_id,
                command_id,
            )

            exit_code = getattr(
                remote,
                "exit_code",
                None,
            )

            if isinstance(exit_code, int):
                command_completed = True
                break

            remaining = (
                deadline - loop.time()
            )

            if remaining <= 0:
                raise TimeoutError(
                    "Daytona control-plane command "
                    f"timed out after {timeout} seconds"
                )

            await asyncio.sleep(
                min(
                    DAYTONA_COMMAND_POLL_INTERVAL_SECONDS,
                    remaining,
                )
            )

        logs = await asyncio.to_thread(
            process.get_session_command_logs,
            session_id,
            command_id,
        )

        result = SandboxCommandResult(
            exit_code=exit_code,
            output=_clean_logs(logs),
        )

    except BaseException as exc:
        if session_created:
            try:
                await _cleanup_session(
                    process,
                    session_id=session_id,
                    pid_file=pid_file,
                    pgid=pgid,
                    terminate=(
                        launch_attempted
                        and not command_completed
                    ),
                )

            except Exception as cleanup_exc:
                add_note = getattr(
                    exc,
                    "add_note",
                    None,
                )

                if callable(add_note):
                    add_note(
                        "Daytona cleanup also failed: "
                        f"{cleanup_exc}"
                    )

        raise

    else:
        await _cleanup_session(
            process,
            session_id=session_id,
            pid_file=pid_file,
            pgid=pgid,
            terminate=False,
        )

        return result


async def execute_control_plane_command(
    sandbox_backend: SandboxBackendProtocol,
    command: str,
    *,
    timeout: int = 60,
    sandbox_type: str | None = None,
) -> SandboxCommandResult:
    """Execute one deterministic runtime-owned sandbox command.

    Daytona commands run in isolated process groups so cancellation
    and runtime timeouts can terminate the entire remote process tree.
    """
    if timeout <= 0:
        raise ValueError(
            "Control-plane command timeout "
            "must be positive"
        )

    provider = (
        sandbox_type
        or os.getenv(
            "SANDBOX_TYPE",
            "langsmith",
        )
    ).strip().lower()

    if provider == "daytona":
        return await _execute_daytona_command(
            sandbox_backend,
            command,
            timeout=timeout,
        )

    result = await sandbox_backend.aexecute(
        command,
        timeout=timeout,
    )

    exit_code = getattr(
        result,
        "exit_code",
        None,
    )

    output: Any = getattr(
        result,
        "output",
        "",
    )

    if not isinstance(exit_code, int):
        raise RuntimeError(
            "Sandbox command returned "
            "an invalid exit code"
        )

    return SandboxCommandResult(
        exit_code=exit_code,
        output=(
            output
            if isinstance(output, str)
            else str(output or "")
        ),
    )
