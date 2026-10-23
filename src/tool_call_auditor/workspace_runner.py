from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path

from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig, ToolsExecutionLoop, ToolsExecutor
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.benchmark import (
    benchmark_suite_with_injections,
    benchmark_suite_without_injections,
    run_task_with_injection_tasks,
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

    benign = len(config.user_tasks) * config.repetitions * len(config.decisions)
    attacked = benign * len(config.injection_tasks)
    attack_goal_checks = (
        len(config.injection_tasks) * config.repetitions * len(config.decisions)
    )
    return {
        "benchmark_version": config.benchmark_version,
        "suite": config.suite,
        "provider": config.provider,
        "model": config.model,
        "attack": config.attack,
        "user_tasks": config.user_tasks,
        "injection_tasks": config.injection_tasks,
        "decisions": config.decisions,
        "repetitions": config.repetitions,
        "planned_benign_episodes": benign,
        "planned_attacked_episodes": attacked,
        "planned_primary_episodes": benign + attacked,
        "planned_attack_goal_checks": attack_goal_checks,
        "planned_total_agent_runs": benign + attacked + attack_goal_checks,
    }


def _replace_tools_executor(
    pipeline: AgentPipeline,
    decision: str,
    decision_log: Path,
    consequential_tools: set[str],
    output_format: str,
) -> None:
    formatter = partial(tool_result_to_str, dump_fn=json.dumps) if output_format == "json" else tool_result_to_str
    for element in pipeline.elements:
        if not isinstance(element, ToolsExecutionLoop):
            continue
        element.elements = [
            AuditedToolsExecutor(decision, decision_log, consequential_tools, formatter)
            if isinstance(child, ToolsExecutor)
            else child
            for child in element.elements
        ]


def build_pipeline(config: PilotConfig, decision: str, decision_log: Path) -> AgentPipeline:
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
    )
    return pipeline


def _jsonable_results(results: dict) -> dict:
    return {
        section: {"|".join(key) if isinstance(key, tuple) else key: value for key, value in values.items()}
        for section, values in results.items()
    }


def run(config: PilotConfig) -> list[dict]:
    validate_config(config, require_model=True)
    suite = get_suite(config.benchmark_version, config.suite)
    summaries = []
    for repetition in range(1, config.repetitions + 1):
        for decision in config.decisions:
            run_dir = config.output_dir / f"repeat-{repetition}" / decision.lower()
            run_dir.mkdir(parents=True, exist_ok=True)
            decision_log = run_dir / "tool_call_decisions.jsonl"
            decision_log.unlink(missing_ok=True)
            pipeline = build_pipeline(config, decision, decision_log)

            with OutputLogger(str(run_dir)):
                benign = benchmark_suite_without_injections(
                    pipeline,
                    suite,
                    user_tasks=config.user_tasks,
                    logdir=run_dir,
                    force_rerun=True,
                    benchmark_version=config.benchmark_version,
                )
                attack = load_attack(config.attack, suite, pipeline)
                attacked = benchmark_suite_with_injections(
                    pipeline,
                    suite,
                    attack,
                    user_tasks=config.user_tasks,
                    injection_tasks=config.injection_tasks,
                    logdir=run_dir,
                    force_rerun=True,
                    benchmark_version=config.benchmark_version,
                )

            summary = {
                "repetition": repetition,
                "decision": decision,
                "benign": _jsonable_results(benign),
                "attacked": _jsonable_results(attacked),
                "decision_log": str(decision_log),
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            summaries.append(summary)
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
        "--smoke-test", action="store_true", help="Run only the first benign task with fixed ALLOW."
    )
    parser.add_argument(
        "--attacked-smoke-test",
        action="store_true",
        help="Run user_task_33 with injection_task_0 under ALLOW and BLOCK.",
    )
    args = parser.parse_args()

    load_dotenv(Path.cwd() / ".env")
    config = PilotConfig.from_yaml(args.config)
    if args.model:
        config.model = args.model
    plan = validate_config(config, require_model=not args.dry_run)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.smoke_test:
        print(json.dumps(run_benign_smoke(config), indent=2))
        return
    if args.attacked_smoke_test:
        print(json.dumps(run_attacked_smoke(config), indent=2))
        return
    summaries = run(config)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
