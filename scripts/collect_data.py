from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

from tool_call_auditor.config import PilotConfig
from tool_call_auditor.data_export import export_dataset
from tool_call_auditor.workspace_runner import run, to_jsonable, validate_config


def _plan_batches(plan_path: Path) -> tuple[dict, list[tuple[str, PilotConfig]]]:
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or not isinstance(plan.get("batches"), list):
        raise ValueError("Expansion plan must contain a batches list")
    common = {
        key: value
        for key, value in plan.items()
        if key not in {"plan_id", "batches"}
    }
    batch_ids = [batch.get("id") for batch in plan["batches"]]
    if any(not isinstance(batch_id, str) or not batch_id for batch_id in batch_ids):
        raise ValueError("Every plan batch must have a non-empty id")
    if len(batch_ids) != len(set(batch_ids)):
        raise ValueError("Expansion plan contains duplicate batch ids")
    loaded = []
    for batch in plan["batches"]:
        config_values = {**common, **{key: value for key, value in batch.items() if key != "id"}}
        config_values["source_id"] = batch.get(
            "source_id", f"{plan.get('source_id', plan.get('plan_id'))}:{batch['id']}"
        )
        loaded.append((batch["id"], PilotConfig.model_validate(config_values)))
    return plan, loaded


def _require_fresh_paths(
    config: PilotConfig,
    allow_existing_output: bool,
    allow_existing_export: bool = False,
) -> None:
    paths = [
        ("output", config.output_dir, allow_existing_output),
        ("export", config.export_dir, allow_existing_export),
    ]
    for label, path, allow_existing in paths:
        if path is None:
            continue
        if allow_existing or not path.exists():
            continue
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(
                f"Plan {label} directory {path} already contains data; "
                "select a new batch path rather than overwriting it."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect and export AgentDojo tool-call and episode data."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--plan",
        type=Path,
        help="Load an immutable batch from an expansion plan YAML file.",
    )
    parser.add_argument(
        "--batch",
        help="Batch id in --plan. Omit only for a plan-wide dry run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the experiment matrix without creating files or calling an API.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export existing AgentDojo traces and audit logs without running an agent.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit existing output/export directories; existing exported files are replaced.",
    )
    parser.add_argument("--resume-run-id", help="Resume a recorded run without rerunning completed tasks.")
    parser.add_argument(
        "--remaining-tasks-file",
        type=Path,
        help="Run only explicit remaining-work selectors from a JSON array.",
    )
    args = parser.parse_args()

    if args.plan and args.config:
        parser.error("Use either --plan or --config, not both")
    if args.plan:
        plan, batches = _plan_batches(args.plan)
        if args.batch:
            selected = [(batch_id, config) for batch_id, config in batches if batch_id == args.batch]
            if not selected:
                parser.error(f"Unknown batch {args.batch!r} in {args.plan}")
            batches = selected
        elif not args.dry_run:
            parser.error("--batch is required when executing or exporting a planned batch")
        elif args.export_only:
            parser.error("--batch is required with --export-only")

        if args.overwrite and not args.dry_run:
            parser.error("Immutable expansion-plan batches do not support --overwrite")
        if args.resume_run_id and args.overwrite:
            parser.error("--resume-run-id cannot be combined with --overwrite")

        batch_reports = []
        for batch_id, config in batches:
            if args.resume_run_id and len(batches) != 1:
                parser.error("Resume one planned batch at a time")
            preflight = validate_config(config, require_model=not args.dry_run)
            export_dir = config.export_dir or config.output_dir / "data"
            preflight.update(
                {
                    "plan_id": plan["plan_id"],
                    "batch_id": batch_id,
                    "source_id": config.source_id,
                    "output_dir": str(config.output_dir),
                    "export_dir": str(export_dir),
                    "api_calls": 0 if args.dry_run else None,
                }
            )
            batch_reports.append(preflight)

        if args.dry_run:
            first_pass_runs = sum(
                report["planned_total_agent_runs"] for report in batch_reports
            )
            max_attempt_runs = sum(
                report["planned_total_agent_runs"] * report["max_episode_attempts"]
                for report in batch_reports
            )
            totals = {
                "planned_primary_episodes": sum(
                    report["planned_primary_episodes"] for report in batch_reports
                ),
                "planned_attack_goal_checks": sum(
                    report["planned_attack_goal_checks"] for report in batch_reports
                ),
                "planned_total_agent_runs": sum(
                    report["planned_total_agent_runs"] for report in batch_reports
                ),
                "max_episode_attempts": max(
                    report["max_episode_attempts"] for report in batch_reports
                ),
                "max_possible_agent_runs_with_retries": max_attempt_runs,
                "additional_retry_run_budget": max_attempt_runs - first_pass_runs,
                "unique_user_tasks": sorted(
                    {task_id for report in batch_reports for task_id in report["user_tasks"]}
                ),
                "attack_coverage": sorted(
                    {report["attack"] for report in batch_reports if report["planned_attacked_episodes"]}
                ),
                "injection_task_coverage": sorted(
                    {
                        pair["injection_task_id"]
                        for report in batch_reports
                        for pair in report["attacked_task_pairs"]
                    }
                ),
            }
            print(
                json.dumps(
                    {"plan_id": plan["plan_id"], "totals": totals, "batches": batch_reports},
                    indent=2,
                )
            )
            return

        batch_id, config = batches[0]
        _require_fresh_paths(config, allow_existing_output=bool(args.resume_run_id))
        export_dir = config.export_dir or config.output_dir / "data"
        if args.export_only:
            paths = export_dataset(config.output_dir, export_dir)
            print(json.dumps({"batch_id": batch_id, "exported_files": [str(path) for path in paths]}, indent=2))
            return

        load_dotenv(Path.cwd() / ".env")
        summaries = run(
            config,
            resume_run_id=args.resume_run_id,
            remaining_tasks_file=args.remaining_tasks_file,
        )
        calls_path, episodes_path = export_dataset(config.output_dir, export_dir)
        summary_path = export_dir / "collection_summary.json"
        print(
            json.dumps(
                to_jsonable(
                    {
                        "batch_id": batch_id,
                        "summaries": summaries,
                        "exported_files": [
                            str(calls_path),
                            str(episodes_path),
                            str(summary_path),
                        ],
                    }
                ),
                indent=2,
            )
        )
        return

    config = PilotConfig.from_yaml(args.config or Path("configs/experiment.yaml"))
    export_dir = config.export_dir or config.output_dir / "data"
    if args.dry_run:
        plan = validate_config(config, require_model=False)
        plan.update(
            {
                "output_dir": str(config.output_dir),
                "export_dir": str(export_dir),
                "api_calls": 0,
            }
        )
        print(json.dumps(plan, indent=2))
        return

    if args.export_only:
        paths = export_dataset(config.output_dir, export_dir, overwrite=args.overwrite)
        print(json.dumps({"exported_files": [str(path) for path in paths]}, indent=2))
        return

    _require_fresh_paths(
        config,
        allow_existing_output=args.resume_run_id is not None or args.overwrite,
        allow_existing_export=args.overwrite,
    )
    load_dotenv(Path.cwd() / ".env")
    run_error: Exception | None = None
    summaries = None
    try:
        summaries = run(
            config,
            overwrite=args.overwrite,
            resume_run_id=args.resume_run_id,
            remaining_tasks_file=args.remaining_tasks_file,
        )
    except Exception as error:
        run_error = error

    try:
        calls_path, episodes_path = export_dataset(
            config.output_dir,
            export_dir,
            overwrite=args.overwrite,
        )
    except Exception as export_error:
        if run_error is not None:
            raise export_error from run_error
        raise
    if run_error is not None:
        raise run_error

    print(
        json.dumps(
            to_jsonable(
                {
                    "summaries": summaries,
                    "exported_files": [str(calls_path), str(episodes_path)],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
