from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from swegen.analyze import classifier as classifier_module
from swegen.analyze.models import Classification
from swegen.create import claude_code_runner as runner_module
from swegen.create import claude_code_utils as cli_module
from swegen.create.claude_code_utils import ClaudeCLIResult


class _FakeStdin:
    def __init__(self) -> None:
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _FakeStream:
    def __init__(self, content: str) -> None:
        self._lines = iter(content.splitlines(keepends=True))

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._lines).encode()
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakeProcess:
    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(stderr)
        self.returncode: int | None = None
        self._final_returncode = returncode

    async def wait(self) -> int:
        self.returncode = self._final_returncode
        return self._final_returncode

    def terminate(self) -> None:
        self.returncode = self._final_returncode

    def kill(self) -> None:
        self.returncode = self._final_returncode


class _HangingProcess(_FakeProcess):
    def __init__(self) -> None:
        super().__init__("")
        self._stopped = asyncio.Event()

    async def wait(self) -> int:
        await self._stopped.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15
        self._stopped.set()

    def kill(self) -> None:
        self.returncode = -9
        self._stopped.set()


def _jsonl(*events: dict[str, Any]) -> str:
    return "".join(json.dumps(event) + "\n" for event in events)


def test_build_command_uses_stream_json_and_structured_output(tmp_path: Path) -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

    command = cli_module.build_claude_command(
        "claude",
        model="opus",
        allowed_tools=["Read", "Glob"],
        add_dirs=[tmp_path],
        output_schema=schema,
    )

    assert command[:5] == ["claude", "--print", "--verbose", "--output-format", "stream-json"]
    assert command[command.index("--model") + 1] == "opus"
    assert command[command.index("--allowedTools") + 1] == "Read,Glob"
    assert "--dangerously-skip-permissions" not in command
    assert "--permission-mode" not in command
    assert command[command.index("--add-dir") + 1] == str(tmp_path.resolve())
    assert json.loads(command[command.index("--json-schema") + 1]) == schema


def test_run_claude_code_streams_prompt_and_parses_result(monkeypatch: Any, tmp_path: Path) -> None:
    process = _FakeProcess(
        _jsonl(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Working"}]},
            },
            {
                "type": "result",
                "result": "Done",
                "structured_output": {"answer": "yes"},
                "is_error": False,
            },
        )
    )
    captured: dict[str, Any] = {}

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> _FakeProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(cli_module, "find_claude_executable", lambda: "claude")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    result = asyncio.run(
        cli_module.run_claude_code(
            "a prompt too large for a command-line argument",
            cwd=tmp_path,
            model="opus",
            allowed_tools=["Read"],
            timeout=10,
        )
    )

    assert process.stdin.data.decode() == "a prompt too large for a command-line argument"
    assert process.stdin.closed
    assert "a prompt too large" not in captured["command"]
    assert captured["kwargs"]["cwd"] == str(tmp_path.resolve())
    assert result.succeeded
    assert result.assistant_text == "Working"
    assert result.result == "Done"
    assert result.structured_output == {"answer": "yes"}


def test_run_claude_code_surfaces_nonzero_exit(monkeypatch: Any, tmp_path: Path) -> None:
    process = _FakeProcess("", stderr="authentication failed\n", returncode=1)

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> _FakeProcess:
        return process

    monkeypatch.setattr(cli_module, "find_claude_executable", lambda: "claude")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    result = asyncio.run(
        cli_module.run_claude_code(
            "prompt",
            cwd=tmp_path,
            model="opus",
            allowed_tools=[],
            timeout=10,
        )
    )

    assert not result.succeeded
    assert result.returncode == 1
    assert result.error_message == "authentication failed"


def test_run_claude_code_preserves_plain_stdout_failure(monkeypatch: Any, tmp_path: Path) -> None:
    process = _FakeProcess("startup failed\n", returncode=1)

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> _FakeProcess:
        return process

    monkeypatch.setattr(cli_module, "find_claude_executable", lambda: "claude")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    result = asyncio.run(
        cli_module.run_claude_code(
            "prompt",
            cwd=tmp_path,
            model="opus",
            allowed_tools=[],
            timeout=10,
        )
    )

    assert not result.succeeded
    assert result.error_message == "startup failed"


def test_run_claude_code_reports_missing_executable(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(cli_module, "find_claude_executable", lambda: None)

    with pytest.raises(cli_module.ClaudeCLINotFoundError, match="not found"):
        asyncio.run(
            cli_module.run_claude_code(
                "prompt",
                cwd=tmp_path,
                model="opus",
                allowed_tools=[],
                timeout=10,
            )
        )


def test_run_claude_code_terminates_on_timeout(monkeypatch: Any, tmp_path: Path) -> None:
    captured: dict[str, _HangingProcess] = {}

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> _HangingProcess:
        process = _HangingProcess()
        captured["process"] = process
        return process

    monkeypatch.setattr(cli_module, "find_claude_executable", lambda: "claude")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    with pytest.raises(cli_module.ClaudeCLITimeoutError, match="timed out"):
        asyncio.run(
            cli_module.run_claude_code(
                "prompt",
                cwd=tmp_path,
                model="opus",
                allowed_tools=[],
                timeout=0,
            )
        )

    assert captured["process"].returncode == -15


def test_classifier_uses_cli_structured_output(monkeypatch: Any, tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    task_dir = tmp_path / "task"
    trial_dir.mkdir()
    task_dir.mkdir()
    (trial_dir / "result.json").write_text(json.dumps({"reward": 1.0}))
    captured: dict[str, Any] = {}

    async def fake_run_claude_code(prompt: str, **kwargs: Any) -> ClaudeCLIResult:
        captured.update(kwargs)
        return ClaudeCLIResult(
            returncode=0,
            result="Emitted.",
            structured_output={
                "classification": "GOOD_SUCCESS",
                "subtype": "Correct Solution",
                "evidence": "The verifier passed.",
                "root_cause": "The implementation is correct.",
                "recommendation": "N/A",
            },
            assistant_text="",
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(classifier_module, "run_claude_code", fake_run_claude_code)
    classifier = classifier_module.TrialClassifier()

    result = asyncio.run(classifier.classify_trial(trial_dir, task_dir))

    assert result.classification == Classification.GOOD_SUCCESS
    assert result.reward == 1.0
    assert captured["cwd"] == trial_dir
    assert captured["add_dirs"] == [task_dir]
    assert captured["allowed_tools"] == ["Read", "Glob"]
    assert captured["output_schema"]["type"] == "object"


def test_generation_runner_invokes_cli(monkeypatch: Any, tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks"
    task_dir = dataset_path / "owner__repo-42"
    repo_path = tmp_path / "repo"
    task_dir.mkdir(parents=True)
    repo_path.mkdir()
    captured: dict[str, Any] = {}

    async def fake_run_claude_code(prompt: str, **kwargs: Any) -> ClaudeCLIResult:
        captured["prompt"] = prompt
        captured.update(kwargs)
        return ClaudeCLIResult(
            returncode=0,
            result="Done",
            structured_output=None,
            assistant_text="",
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(runner_module, "run_claude_code", fake_run_claude_code)

    result = asyncio.run(
        runner_module._run_claude_code_session_async(
            repo="owner/repo",
            pr_number=42,
            repo_path=repo_path,
            task_dir=task_dir,
            task_id=task_dir.name,
            dataset_path=dataset_path,
            test_files=["tests/test_feature.py"],
            timeout=10,
        )
    )

    assert not result.success  # No fake Harbor job results were created.
    assert captured["model"] == "claude-opus-4-8"
    assert captured["allowed_tools"] == ["Read", "Write", "Edit", "Glob", "Grep", "LS", "Bash"]
    assert "owner/repo" in captured["prompt"]
    assert "tests/test_feature.py" in captured["prompt"]
