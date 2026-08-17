from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ANSI color codes for verbose output
class Colors:
    BLUE = "\033[94m"  # Assistant messages
    CYAN = "\033[96m"  # Tool use
    MAGENTA = "\033[95m"  # Tool results
    GREEN = "\033[92m"  # Success/final result
    YELLOW = "\033[93m"  # System messages
    RED = "\033[91m"  # Errors
    BOLD = "\033[1m"
    RESET = "\033[0m"


class ClaudeCLIError(RuntimeError):
    """Base error raised while invoking the Claude Code CLI."""


class ClaudeCLINotFoundError(ClaudeCLIError):
    """Raised when the Claude Code executable cannot be found."""


class ClaudeCLITimeoutError(ClaudeCLIError):
    """Raised after a Claude Code subprocess exceeds its time limit."""


@dataclass(frozen=True)
class ClaudeCLIResult:
    """Parsed result of a non-interactive Claude Code subprocess."""

    returncode: int
    result: str | None
    structured_output: Any
    assistant_text: str
    stdout: str
    stderr: str
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and self.error_message is None


EventCallback = Callable[[dict[str, Any]], None]


def find_claude_executable() -> str | None:
    """Return the Claude Code executable available on PATH."""
    return shutil.which("claude")


def get_claude_version() -> str | None:
    """Return `claude --version`, or None when Claude Code is unavailable."""
    executable = find_claude_executable()
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or result.stderr.strip() or None


def build_claude_command(
    executable: str,
    *,
    model: str,
    allowed_tools: Sequence[str],
    add_dirs: Sequence[Path] = (),
    output_schema: Mapping[str, Any] | None = None,
) -> list[str]:
    """Build a non-interactive Claude Code command.

    The prompt is intentionally sent on stdin by `run_claude_code`, not as a command-line
    argument. SWE-gen prompts are large enough to exceed Windows' command-line length limit.
    """
    command = [
        executable,
        "--print",
        "--verbose",
        "--output-format",
        "stream-json",
        "--model",
        model,
    ]
    if allowed_tools:
        command.extend(["--allowedTools", ",".join(allowed_tools)])
    if add_dirs:
        command.append("--add-dir")
        command.extend(str(path.resolve()) for path in add_dirs)
    if output_schema is not None:
        command.extend(["--json-schema", json.dumps(output_schema, separators=(",", ":"))])
    return command


def _content_blocks(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    message = event.get("message")
    if not isinstance(message, Mapping):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _event_error(event: Mapping[str, Any]) -> str | None:
    event_type = event.get("type")
    if event_type == "error":
        error = event.get("error")
        if isinstance(error, Mapping):
            return str(error.get("message") or error)
        return str(error or event.get("message") or "Claude Code reported an error")
    if event_type == "result" and event.get("is_error"):
        return str(event.get("result") or event.get("error") or "Claude Code reported an error")
    return None


def print_stream_event(event: Mapping[str, Any]) -> None:
    """Print a Claude Code stream-json event with concise colored formatting."""
    event_type = event.get("type")

    if event_type == "assistant":
        for block in _content_blocks(event):
            block_type = block.get("type")
            if block_type == "text":
                text = str(block.get("text") or "")
                if text.strip():
                    print(f"\n{Colors.BLUE}[Assistant]{Colors.RESET} {text}", flush=True)
            elif block_type == "tool_use":
                tool_name = str(block.get("name") or "unknown").upper()
                tool_input = block.get("input", {})
                summary: dict[str, Any] | str
                if isinstance(tool_input, dict):
                    max_len = 2000 if tool_name.lower() == "bash" else 1000
                    summary = {
                        str(key): (
                            value[:max_len] + "..."
                            if isinstance(value, str) and len(value) > max_len
                            else value
                        )
                        for key, value in tool_input.items()
                    }
                else:
                    summary = str(tool_input)[:2000]
                print(
                    f"\n{Colors.CYAN}{Colors.BOLD}{tool_name}{Colors.RESET}: {summary}",
                    flush=True,
                )

    elif event_type == "user":
        for block in _content_blocks(event):
            if block.get("type") != "tool_result":
                continue
            content = block.get("content", "")
            if isinstance(content, str) and len(content) > 2000:
                content = content[:2000] + f"... ({len(content)} chars total)"
            print(f"{Colors.MAGENTA}[Tool Result]{Colors.RESET} {content}", flush=True)

    elif event_type == "result":
        result_text = str(event.get("result") or "")
        if result_text.strip():
            if len(result_text) > 3000:
                result_text = result_text[:3000] + f"... ({len(result_text)} chars total)"
            print(
                f"\n{Colors.GREEN}{Colors.BOLD}[Final Result]{Colors.RESET}\n{result_text}",
                flush=True,
            )
        cost = event.get("total_cost_usd")
        if cost is not None:
            print(f"{Colors.GREEN}[Cost]{Colors.RESET} ${cost}", flush=True)

    elif event_type == "system":
        message = event.get("message") or event.get("subtype")
        if message:
            print(f"{Colors.YELLOW}[System]{Colors.RESET} {message}", flush=True)

    error = _event_error(event)
    if error:
        print(f"{Colors.RED}[Error]{Colors.RESET} {error}", flush=True)


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        process.kill()
        await process.wait()


async def run_claude_code(
    prompt: str,
    *,
    cwd: Path,
    model: str,
    allowed_tools: Sequence[str],
    timeout: int,
    add_dirs: Sequence[Path] = (),
    output_schema: Mapping[str, Any] | None = None,
    verbose: bool = False,
    event_callback: EventCallback | None = None,
    env: Mapping[str, str] | None = None,
) -> ClaudeCLIResult:
    """Run Claude Code in print mode and parse its stream-json output.

    Stdout and stderr are drained concurrently so a verbose Claude session cannot deadlock
    on a full pipe. The implementation uses asyncio subprocesses for the same behavior on
    Windows and POSIX.
    """
    executable = find_claude_executable()
    if executable is None:
        raise ClaudeCLINotFoundError(
            "Claude Code CLI not found. Install it from https://claude.ai/code and ensure "
            "the `claude` command is on PATH."
        )

    command = build_claude_command(
        executable,
        model=model,
        allowed_tools=allowed_tools,
        add_dirs=add_dirs,
        output_schema=output_schema,
    )

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd.resolve()),
            env=dict(env) if env is not None else os.environ.copy(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=10 * 1024 * 1024,
        )
    except FileNotFoundError as exc:
        raise ClaudeCLINotFoundError(f"Claude Code CLI could not be started: {exc}") from exc
    except OSError as exc:
        raise ClaudeCLIError(f"Claude Code CLI could not be started: {exc}") from exc

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    assistant_parts: list[str] = []
    result_text: str | None = None
    structured_output: Any = None
    event_error: str | None = None

    async def write_prompt() -> None:
        assert process.stdin is not None
        process.stdin.write(prompt.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()
        await process.stdin.wait_closed()

    async def read_stdout() -> None:
        nonlocal event_error, result_text, structured_output
        assert process.stdout is not None
        async for raw_line in process.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            stdout_lines.append(line)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if verbose and line:
                    print(line, flush=True)
                continue
            if not isinstance(event, dict):
                continue
            if verbose:
                print_stream_event(event)
            if event_callback is not None:
                event_callback(event)
            for block in _content_blocks(event):
                if block.get("type") == "text":
                    assistant_parts.append(str(block.get("text") or ""))
            if event.get("type") == "result":
                raw_result = event.get("result")
                result_text = str(raw_result) if raw_result is not None else None
                structured_output = event.get("structured_output")
            event_error = event_error or _event_error(event)

    async def read_stderr() -> None:
        assert process.stderr is not None
        async for raw_line in process.stderr:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            stderr_lines.append(line)
            if verbose and line:
                print(f"[stderr] {line}", file=sys.stderr, flush=True)

    async def communicate() -> None:
        await asyncio.gather(write_prompt(), read_stdout(), read_stderr(), process.wait())

    try:
        await asyncio.wait_for(communicate(), timeout=timeout)
    except TimeoutError as exc:
        await _stop_process(process)
        raise ClaudeCLITimeoutError(f"Claude Code timed out after {timeout} seconds") from exc
    except (BrokenPipeError, ConnectionResetError) as exc:
        await _stop_process(process)
        raise ClaudeCLIError(f"Claude Code closed its input unexpectedly: {exc}") from exc

    stdout = "\n".join(stdout_lines)
    stderr = "\n".join(stderr_lines)
    returncode = process.returncode if process.returncode is not None else -1
    error_message = event_error
    if returncode != 0 and error_message is None:
        error_message = (
            stderr.strip()
            or result_text
            or stdout.strip()
            or f"Claude Code exited with code {returncode}"
        )

    return ClaudeCLIResult(
        returncode=returncode,
        result=result_text,
        structured_output=structured_output,
        assistant_text="\n".join(part for part in assistant_parts if part),
        stdout=stdout,
        stderr=stderr,
        error_message=error_message,
    )
