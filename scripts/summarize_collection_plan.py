from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _merge_counts(target: dict[str, int], values: dict[str, int]) -> None:
    for key, value in values.items():
        target[key] = target.get(key, 0) + value


def summarize_plan(plan_path: Path, output_path: Path | None = None) -> Path:
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or not isinstance(plan.get("batches"), list):
        raise ValueError("Collection plan must contain a batches list")
    destination = output_path or Path("data/exports/real-traces-phase-1/phase_summary.json")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing phase summary: {destination}")

    summaries = []
    for batch in plan["batches"]:
        batch_id = batch["id"]
        export_dir = Path(batch["export_dir"])
        summary_path = export_dir / "collection_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Batch {batch_id} has no exported collection summary: {summary_path}"
            )
        summaries.append(
            {
                "batch_id": batch_id,
                "summary": json.loads(summary_path.read_text(encoding="utf-8")),
            }
        )

    total = {
        "plan_id": plan.get("plan_id"),
        "batch_count": len(summaries),
        "primary_episode_count": 0,
        "attack_goal_check_count": 0,
        "candidate_consequential_call_count": 0,
        "provider_api_request_count": 0,
        "provider_api_request_count_observed_batches": 0,
        "task_group_coverage": set(),
        "attack_coverage": {},
        "tool_coverage": {},
        "injection_exposure": {
            "requested_attack_episode_count": 0,
            "injection_task_coverage": {},
            "confirmed_exposure_count": 0,
            "not_observed_count": 0,
            "confirmed_by_injection_task": {},
        },
        "errors": {"failed_task_attempt_count": 0, "failed_episode_count": 0},
        "duration_seconds": {
            "episode_sum": 0.0,
            "episode_count": 0,
            "api_request_duration_sum_seconds": 0.0,
            "episodes_with_api_request_duration": 0,
            "task_attempt_sum": 0.0,
        },
        "token_usage": {},
        "source_ids": set(),
        "labels_all_null": True,
        "batches": [],
    }

    token_values: dict[str, list[int | None]] = {}
    for entry in summaries:
        batch_id, summary = entry["batch_id"], entry["summary"]
        total["batches"].append(
            {
                "batch_id": batch_id,
                "primary_episode_count": summary["primary_episode_count"],
                "attack_goal_check_count": summary["attack_goal_check_count"],
                "candidate_consequential_call_count": summary[
                    "candidate_consequential_call_count"
                ],
            }
        )
        for field in (
            "primary_episode_count",
            "attack_goal_check_count",
            "candidate_consequential_call_count",
            "provider_api_request_count",
        ):
            value = summary.get(field)
            if isinstance(value, int):
                total[field] += value
                if field == "provider_api_request_count":
                    total["provider_api_request_count_observed_batches"] += 1
        total["task_group_coverage"].update(summary.get("task_group_coverage", []))
        total["source_ids"].update(summary.get("source_ids", []))
        _merge_counts(total["attack_coverage"], summary.get("attack_coverage", {}))
        _merge_counts(total["tool_coverage"], summary.get("tool_coverage", {}))

        exposure = summary.get("injection_exposure", {})
        for field in (
            "requested_attack_episode_count",
            "confirmed_exposure_count",
            "not_observed_count",
        ):
            total["injection_exposure"][field] += exposure.get(field, 0)
        _merge_counts(
            total["injection_exposure"]["injection_task_coverage"],
            exposure.get("injection_task_coverage", {}),
        )
        _merge_counts(
            total["injection_exposure"]["confirmed_by_injection_task"],
            exposure.get("confirmed_by_injection_task", {}),
        )
        _merge_counts(total["errors"], summary.get("errors", {}))
        durations = summary.get("duration_seconds", {})
        for field in (
            "episode_sum",
            "api_request_duration_sum_seconds",
            "task_attempt_sum",
        ):
            value = durations.get(field)
            if field == "api_request_duration_sum_seconds" and value is None:
                value = durations.get("api_request_sum")
            if isinstance(value, (int, float)):
                total["duration_seconds"][field] += value
        total["duration_seconds"]["episode_count"] += durations.get("episode_count", 0)
        total["duration_seconds"]["episodes_with_api_request_duration"] += durations.get(
            "episodes_with_api_request_duration",
            durations.get("api_request_count", 0),
        )
        for field, value in summary.get("token_usage", {}).items():
            if field.endswith("_known_episode_count") or field.endswith("_missing_episode_count"):
                total["token_usage"][field] = total["token_usage"].get(field, 0) + value
                continue
            if field in {"prompt_tokens", "completion_tokens", "total_tokens"}:
                token_values.setdefault(field, []).append(value)
        total["labels_all_null"] = (
            total["labels_all_null"] and summary.get("labels", {}).get("all_null", False)
        )

    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = token_values.get(field, [])
        total["token_usage"][field] = (
            sum(value for value in values if value is not None)
            if values and all(value is not None for value in values)
            else None
        )
        total["token_usage"][f"{field}_observed_batches"] = sum(
            value is not None for value in values
        )
        total["token_usage"][f"{field}_missing_batches"] = len(values) - sum(
            value is not None for value in values
        )
    duration = total["duration_seconds"]
    duration["episode_mean"] = (
        duration["episode_sum"] / duration["episode_count"]
        if duration["episode_count"]
        else None
    )
    duration["mean_api_request_duration_seconds_per_episode"] = (
        duration["api_request_duration_sum_seconds"]
        / duration["episodes_with_api_request_duration"]
        if duration["episodes_with_api_request_duration"]
        else None
    )
    total["task_group_coverage"] = sorted(total["task_group_coverage"])
    if not total["provider_api_request_count_observed_batches"]:
        total["provider_api_request_count"] = None
    total["source_ids"] = sorted(total["source_ids"])
    total["attack_coverage"] = dict(sorted(total["attack_coverage"].items()))
    total["tool_coverage"] = dict(sorted(total["tool_coverage"].items()))
    for field in ("injection_task_coverage", "confirmed_by_injection_task"):
        total["injection_exposure"][field] = dict(
            sorted(total["injection_exposure"][field].items())
        )
    total["batches"] = summaries

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(total, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate real collection summaries without changing batch exports."
    )
    parser.add_argument("--plan", type=Path, default=Path("configs/real_trace_expansion_plan.yaml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(summarize_plan(args.plan, args.output))


if __name__ == "__main__":
    main()
