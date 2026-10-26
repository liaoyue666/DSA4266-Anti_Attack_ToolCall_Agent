from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig, ToolsExecutionLoop, ToolsExecutor
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.benchmark import (
    benchmark_suite_without_injections,
    run_task_with_injection_tasks,
    run_task_without_injection_tasks,
)
from agentdojo.logging import OutputLogger
from agentdojo.models import MODEL_NAMES, ModelsEnum
from agentdojo.task_suite import get_suite, get_suites
from openai import OpenAI
from dotenv import load_dotenv

from .config import PilotConfig
from .deepseek_llm import DeepSeekLLM
from .executor import AuditedToolsExecutor

# AgentDojo's model-addressed attacks need a prose family name. Version 0.1.35
# predates DeepSeek support, so register the selected model explicitly.
MODEL_NAMES.setdefault("deepseek-flash", "DeepSeek")


def validate_config(config: PilotConfig, require_model: bool) -> dict:
    available_suites = get_suites(config.benchmark_version)
    if config.suite not in available_suites:
        raise ValueError(f"Unknown suite {config.suite!r}; choose from {sorted(available_suites)}")
    suite = get_suite(config.benchmark_version, config.suite)

    missing_users = set(config.user_tasks) - set(suite.user_tasks)
    missing_injections = set(config.injection_tasks) - set(suite.injection_tasks)
    missing_tools = set(config.consequential_tools) - {tool.name for tool in suite.tools}
    if missing_users:
        raise ValueError(f"Unknown user tasks: {sorted(missing_users)}")
    if missing_injections:
        raise ValueError(f"Unknown injection tasks: {sorted(missing_injections)}")
    if missing_tools:
        raise ValueError(f"Unknown consequential tools: {sorted(missing_tools)}")
    if config.attack not in ATTACKS:
        raise ValueError(f"Unknown attack {config.attack!r}; choose from {sorted(ATTACKS)}")
    if require_model and config.model is None:
        raise ValueError("Set 'model' in the YAML config or pass --model before a real run")
    if (
        config.provider == "agentdojo"
        and config.model is not None
        and config.model not in {str(model) for model in ModelsEnum}
    ):
        raise ValueError(f"Model {config.model!r} is not supported by AgentDojo 0.1.35")
    if config.provider == "deepseek" and config.model != "deepseek-flash":
        raise ValueError("This pilot currently pins the DeepSeek model to 'deepseek-flash'")

    if not config.consequential_tools:
        raise ValueError("consequential_tools must name at least one tool explicitly")
    pairs = _attacked_pairs(config)
    runs_per_condition = config.repetitions * len(config.decisions)
    has_benign = config.run_mode in {"both", "benign_only"}
    has_attacked = config.run_mode in {"both", "attacked_only"}
    benign = len(config.user_tasks) * runs_per_condition if has_benign else 0
    attacked = len(pairs) * runs_per_condition if has_attacked else 0
    attack_goal_checks = (
        len(set(config.injection_tasks)) * runs_per_condition
        if has_attacked and config.run_attack_goal_checks
        else 0
    )
    return {
        "benchmark_version": config.benchmark_version,
        "suite": config.suite,
        "provider": config.provider,
        "model": config.model,
        "attack": config.attack,
        "run_mode": config.run_mode,
        "user_tasks": config.user_tasks,
        "injection_tasks": config.injection_tasks,
        "attacked_task_pairs": [
            {"user_task_id": user_task_id, "injection_task_id": injection_task_id}
            for user_task_id, injection_task_id in pairs
        ],
        "decisions": config.decisions,
        "repetitions": config.repetitions,
        "max_episode_attempts": config.max_episode_attempts,
        "planned_benign_episodes": benign,
        "planned_attacked_episodes": attacked,
        "planned_primary_episodes": benign + attacked,
        "planned_attack_goal_checks": attack_goal_checks,
        "planned_total_agent_runs": benign + attacked + attack_goal_checks,
    }


def _attacked_pairs(config: PilotConfig) -> list[tuple[str, str]]:
    if config.run_mode == "benign_only":
        return []
    if config.attacked_task_pairs is not None:
        return list(config.attacked_task_pairs)
    return [
        (user_task_id, injection_task_id)
        for user_task_id in config.user_tasks
        for injection_task_id in config.injection_tasks
    ]


def _replace_tools_executor(
    pipeline: AgentPipeline,
    decision: str,
    decision_log: Path,
    consequential_tools: set[str],
    output_format: str,
    run_id: str | None = None,
    repetition: int | None = None,
) -> None:
    formatter = partial(tool_result_to_str, dump_fn=json.dumps) if output_format == "json" else tool_result_to_str
    for element in pipeline.elements:
        if not isinstance(element, ToolsExecutionLoop):
            continue
        element.elements = [
            AuditedToolsExecutor(
                decision,
                decision_log,
                consequential_tools,
                formatter,
                run_id=run_id,
                repetition=repetition,
            )
            if isinstance(child, ToolsExecutor)
            else child
            for child in element.elements
        ]


def build_pipeline(
    config: PilotConfig,
    decision: str,
    decision_log: Path,
    run_id: str | None = None,
    repetition: int | None = None,
    telemetry_path: Path | None = None,
) -> AgentPipeline:
    assert config.model is not None
    if config.provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for a real DeepSeek run")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        llm = DeepSeekLLM(
            OpenAI(api_key=api_key, base_url=base_url),
            model=config.model,
            temperature=config.temperature,
            run_id=run_id,
            repetition=repetition,
            decision=decision,
            telemetry_path=telemetry_path,
        )
        pipeline_llm = llm
    else:
        pipeline_llm = ModelsEnum(config.model)
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=pipeline_llm,
            model_id=config.model_id,
            defense=None,
            tool_output_format=config.tool_output_format,
            system_message_name=None,
            system_message=None,
        )
    )
    pipeline.name = f"{pipeline.name}-audit-{decision.lower()}"
    _replace_tools_executor(
        pipeline,
        decision,
        decision_log,
        set(config.consequential_tools),
        config.tool_output_format,
        run_id=run_id,
        repetition=repetition,
    )
    return pipeline


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if isinstance(key, tuple):
                json_key = "|".join(str(part) for part in key)
            elif isinstance(key, (str, int, float, bool)) or key is None:
                json_key = key
            else:
                json_key = str(key)
            converted[json_key] = to_jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [to_jsonable(item) for item in sorted(value, key=repr)]
    return value


def _jsonable_results(results: dict) -> dict:
    return to_jsonable(results)


def _work_item_key(
    item: dict, repetition: int, decision: str
) -> str:
    return json.dumps(
        {
            "scenario": item["scenario"],
            "user_task_id": item.get("user_task_id"),
            "injection_task_id": item.get("injection_task_id"),
            "repetition": repetition,
            "decision": decision,
        },
        sort_keys=True,
    )


def _read_status(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _append_status(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def _load_remaining_work(path: Path) -> set[tuple[str, str | None, str | None]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("remaining task file must be a non-empty JSON array")
    selected = set()
    for item in payload:
        if not isinstance(item, dict) or item.get("scenario") not in {
            "benign",
            "attacked",
            "attack_goal_check",
        }:
            raise ValueError("each remaining task must name a supported scenario")
        scenario = item["scenario"]
        user_task_id = item.get("user_task_id")
        injection_task_id = item.get("injection_task_id")
        if scenario == "benign" and (not user_task_id or injection_task_id is not None):
            raise ValueError("benign remaining tasks require only user_task_id")
        if scenario in {"attacked", "attack_goal_check"} and not injection_task_id:
            raise ValueError(f"{scenario} remaining tasks require injection_task_id")
        if scenario == "attacked" and not user_task_id:
            raise ValueError("attacked remaining tasks require user_task_id")
        selected.add((scenario, user_task_id, injection_task_id))
    return selected


def _safe_error_message(error: Exception) -> str:
    message = str(error)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return re.sub(
        r"(?i)(api[_ -]?key|authorization)(\s*[:=]\s*|\s+)[^\s,;]+",
        r"\1\2[REDACTED]",
        message,
    )


def _execute_work_item_attempt(
    config: PilotConfig,
    suite,
    run_id: str,
    run_dir: Path,
    status_path: Path,
    item: dict,
    key: str,
    repetition: int,
    decision: str,
    attempt: int,
) -> tuple[dict, dict | None]:
    attempt_run_id = str(uuid4())
    task_group_id = (
        f"{config.suite}:{item['user_task_id']}" if item.get("user_task_id") else None
    )
    attempt_dir = (
        run_dir
        / f"repeat-{repetition}"
        / decision.lower()
        / "attempts"
        / attempt_run_id
    )
    attempt_dir.mkdir(parents=True, exist_ok=False)
    attempt_manifest_path = attempt_dir / "run_manifest.json"
    attempt_manifest = {
        "run_id": attempt_run_id,
        "parent_run_id": run_id,
        "source_id": config.source_id or run_id,
        "task_group_id": task_group_id,
        "work_item": item,
        "attempt": attempt,
        "status": "running",
        "started_at": time(),
    }
    attempt_manifest_path.write_text(json.dumps(attempt_manifest, indent=2), encoding="utf-8")
    decision_log = attempt_dir / "tool_call_decisions.jsonl"
    status = "completed"
    result = None
    error_record = None
    try:
        pipeline = build_pipeline(
            config,
            decision,
            decision_log,
            run_id=attempt_run_id,
            repetition=repetition,
            telemetry_path=attempt_dir / "api_requests.jsonl",
        )
        with OutputLogger(str(attempt_dir)):
            if item["scenario"] == "benign":
                result = benchmark_suite_without_injections(
                    pipeline,
                    suite,
                    user_tasks=[item["user_task_id"]],
                    logdir=attempt_dir,
                    force_rerun=True,
                    benchmark_version=config.benchmark_version,
                )
            elif item["scenario"] == "attacked":
                attack = load_attack(config.attack, suite, pipeline)
                user_task = suite.get_user_task_by_id(item["user_task_id"])
                utility, security = run_task_with_injection_tasks(
                    suite=suite,
                    agent_pipeline=pipeline,
                    user_task=user_task,
                    attack=attack,
                    logdir=attempt_dir,
                    force_rerun=True,
                    injection_tasks=[item["injection_task_id"]],
                    benchmark_version=config.benchmark_version,
                )
                pair = (item["user_task_id"], item["injection_task_id"])
                result = {"utility": utility.get(pair), "security": security.get(pair)}
            else:
                injection_task = suite.get_injection_task_by_id(item["injection_task_id"])
                result = run_task_without_injection_tasks(
                    suite,
                    pipeline,
                    injection_task,
                    attempt_dir,
                    force_rerun=True,
                    benchmark_version=config.benchmark_version,
                )
    except Exception as error:
        status = "failed"
        error_record = {
            "type": type(error).__name__,
            "message": _safe_error_message(error),
        }

    finished_at = time()
    attempt_manifest.update(
        {
            "status": status,
            "finished_at": finished_at,
            "duration_seconds": finished_at - attempt_manifest["started_at"],
            "error": error_record,
        }
    )
    attempt_manifest_path.write_text(
        json.dumps(attempt_manifest, indent=2), encoding="utf-8"
    )
    status_record = {
        "work_item_key": key,
        **item,
        "task_group_id": task_group_id,
        "repetition": repetition,
        "decision": decision,
        "attempt": attempt,
        "attempt_run_id": attempt_run_id,
        "status": status,
        "duration_seconds": attempt_manifest["duration_seconds"],
        "error": error_record,
    }
    _append_status(status_path, status_record)
    return status_record, result


def run(
    config: PilotConfig,
    overwrite: bool = False,
    resume_run_id: str | None = None,
    remaining_tasks_file: Path | None = None,
) -> list[dict]:
    validate_config(config, require_model=True)
    suite = get_suite(config.benchmark_version, config.suite)
    plan_fingerprint = hashlib.sha256(
        json.dumps(config.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    if resume_run_id:
        run_dir = config.output_dir / "runs" / resume_run_id
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume unknown run: {run_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_fingerprint") != plan_fingerprint:
            raise ValueError("Resume config does not match the original run")
        run_id = resume_run_id
    else:
        if config.output_dir.exists() and any(config.output_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory {config.output_dir} already contains data; "
                "choose a new output_dir or pass --overwrite."
            )
        run_id = str(uuid4())
        run_dir = config.output_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = run_dir / "run_manifest.json"
        manifest = {
            "run_id": run_id,
            "status": "running",
            "started_at": time(),
            "source_id": config.source_id or run_id,
            "config_fingerprint": plan_fingerprint,
            "config": config.model_dump(mode="json"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    work_items = []
    if config.run_mode in {"both", "benign_only"}:
        work_items.extend(
            {"scenario": "benign", "user_task_id": task_id}
            for task_id in config.user_tasks
        )
    if config.run_mode in {"both", "attacked_only"}:
        work_items.extend(
            {
                "scenario": "attacked",
                "user_task_id": user_task_id,
                "injection_task_id": injection_task_id,
            }
            for user_task_id, injection_task_id in _attacked_pairs(config)
        )
        if config.run_attack_goal_checks:
            work_items.extend(
                {"scenario": "attack_goal_check", "injection_task_id": injection_id}
                for injection_id in config.injection_tasks
            )

    if remaining_tasks_file:
        requested = _load_remaining_work(remaining_tasks_file)
        available = {
            (item["scenario"], item.get("user_task_id"), item.get("injection_task_id"))
            for item in work_items
        }
        unknown = requested - available
        if unknown:
            raise ValueError(f"Remaining task file contains unplanned work: {sorted(unknown)}")
        work_items = [
            item
            for item in work_items
            if (
                item["scenario"],
                item.get("user_task_id"),
                item.get("injection_task_id"),
            )
            in requested
        ]

    status_path = run_dir / "task_status.jsonl"
    previous_status = _read_status(status_path)
    completed = {
        record["work_item_key"]
        for record in previous_status
        if record.get("status") == "completed" and record.get("work_item_key")
    }
    attempt_counts: dict[str, int] = {}
    for record in previous_status:
        key = record.get("work_item_key")
        if key:
            attempt_counts[key] = max(attempt_counts.get(key, 0), record.get("attempt", 0))

    summaries = []
    for repetition in range(1, config.repetitions + 1):
        for decision in config.decisions:
            for item in work_items:
                key = _work_item_key(item, repetition, decision)
                if key in completed or attempt_counts.get(key, 0) >= config.max_episode_attempts:
                    continue
                for attempt in range(
                    attempt_counts.get(key, 0) + 1,
                    config.max_episode_attempts + 1,
                ):
                    attempt_counts[key] = attempt
                    status_record, result = _execute_work_item_attempt(
                        config,
                        suite,
                        run_id,
                        run_dir,
                        status_path,
                        item,
                        key,
                        repetition,
                        decision,
                        attempt,
                    )
                    summaries.append({**status_record, "result": to_jsonable(result)})
                    if status_record["status"] == "completed":
                        completed.add(key)
                        break

    failed_work = {
        record["work_item_key"]
        for record in _read_status(status_path)
        if record.get("status") == "failed"
    } - completed
    manifest.update(
        {
            "status": "completed_with_errors" if failed_work else "completed",
            "finished_at": time(),
            "remaining_failed_work_items": len(failed_work),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summaries


def run_benign_smoke(config: PilotConfig) -> dict:
    """Run one benign read-only episode before any full experiment."""
    validate_config(config, require_model=True)
    suite = get_suite(config.benchmark_version, config.suite)
    run_dir = config.output_dir / "smoke" / "allow"
    run_dir.mkdir(parents=True, exist_ok=True)
    decision_log = run_dir / "tool_call_decisions.jsonl"
    decision_log.unlink(missing_ok=True)
    pipeline = build_pipeline(config, "ALLOW", decision_log)
    smoke_task = config.user_tasks[0]
    with OutputLogger(str(run_dir)):
        result = benchmark_suite_without_injections(
            pipeline,
            suite,
            user_tasks=[smoke_task],
            logdir=run_dir,
            force_rerun=True,
            benchmark_version=config.benchmark_version,
        )
    summary = {
        "smoke_test": True,
        "user_task": smoke_task,
        "decision": "ALLOW",
        "result": _jsonable_results(result),
        "decision_log": str(decision_log),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_attacked_smoke(config: PilotConfig) -> list[dict]:
    """Run one attacked task under ALLOW and BLOCK without the full matrix."""
    validate_config(config, require_model=True)
    suite = get_suite(config.benchmark_version, config.suite)
    user_task_id = "user_task_33"
    injection_task_id = "injection_task_0"
    if user_task_id not in config.user_tasks or injection_task_id not in config.injection_tasks:
        raise ValueError("Attacked smoke IDs must remain present in the pilot manifest")
    user_task = suite.get_user_task_by_id(user_task_id)
    summaries = []

    for decision in ("ALLOW", "BLOCK"):
        run_dir = config.output_dir / "attacked-smoke" / decision.lower()
        run_dir.mkdir(parents=True, exist_ok=True)
        decision_log = run_dir / "tool_call_decisions.jsonl"
        decision_log.unlink(missing_ok=True)
        pipeline = build_pipeline(config, decision, decision_log)
        attack = load_attack(config.attack, suite, pipeline)
        with OutputLogger(str(run_dir)):
            utility, security = run_task_with_injection_tasks(
                suite=suite,
                agent_pipeline=pipeline,
                user_task=user_task,
                attack=attack,
                logdir=run_dir,
                force_rerun=True,
                injection_tasks=[injection_task_id],
                benchmark_version=config.benchmark_version,
            )
        key = (user_task_id, injection_task_id)
        summary = {
            "smoke_test": "attacked",
            "user_task": user_task_id,
            "injection_task": injection_task_id,
            "decision": decision,
            "utility": utility[key],
            # AgentDojo's injection task `security()` methods return True when
            # the attack goal is present in the post-execution environment.
            "agentdojo_security_result": security[key],
            "injection_goal_achieved": security[key],
            "attack_succeeded": security[key],
            "decision_log": str(decision_log),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the selected AgentDojo workspace pilot.")
    parser.add_argument("--config", type=Path, default=Path("configs/workspace_pilot.yaml"))
    parser.add_argument("--model", help="Override the AgentDojo model name in the config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the matrix without API calls.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit using an output directory that already contains data.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run only the first benign task with fixed ALLOW."
    )
    parser.add_argument(
        "--attacked-smoke-test",
        action="store_true",
        help="Run user_task_33 with injection_task_0 under ALLOW and BLOCK.",
    )
    parser.add_argument(
        "--resume-run-id",
        help="Continue a run in the configured output_dir, skipping completed work items.",
    )
    parser.add_argument(
        "--remaining-tasks-file",
        type=Path,
        help="Restrict work to an explicit JSON array of remaining task selectors.",
    )
    args = parser.parse_args()

    config = PilotConfig.from_yaml(args.config)
    if args.model:
        config.model = args.model
    plan = validate_config(config, require_model=not args.dry_run)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    load_dotenv(Path.cwd() / ".env")
    if args.smoke_test:
        print(json.dumps(to_jsonable(run_benign_smoke(config)), indent=2))
        return
    if args.attacked_smoke_test:
        print(json.dumps(to_jsonable(run_attacked_smoke(config)), indent=2))
        return
    summaries = run(
        config,
        overwrite=args.overwrite,
        resume_run_id=args.resume_run_id,
        remaining_tasks_file=args.remaining_tasks_file,
    )
    print(json.dumps(to_jsonable(summaries), indent=2))


if __name__ == "__main__":
    main()
