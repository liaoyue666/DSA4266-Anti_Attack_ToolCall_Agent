from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from agentdojo.task_suite import get_suite

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_constructed_candidates import (
    DEFAULT_OBSERVED_EXPORT_ROOTS,
    build_dry_run_report,
    _load_phase_records,
    _model_input_fingerprint,
    _observed_model_input,
    _read_catalog,
)

PLAN_ID = "workspace-real-traces-phase-2"
PRIMARY_PLAN = Path("configs/real_trace_expansion_phase2.yaml")
RESERVE_PLAN = Path("configs/real_trace_expansion_phase2_reserve.yaml")
BANKING_PLAN = Path("configs/banking_compatibility_plan.yaml")
SUMMARY_PATH = Path("data/manifests/phase2_collection_plan.json")
DRY_RUN_PATH = Path("data/manifests/phase2_collection_dry_run.json")
BANKING_SUMMARY_PATH = Path("data/manifests/banking_compatibility_dry_run.json")
MAIN_EPISODE_CAP = 250
BATCH_SIZE = 10
ATTACKS = ("important_instructions", "tool_knowledge")
CONSEQUENTIAL_TOOLS = (
    "send_email",
    "delete_email",
    "create_calendar_event",
    "cancel_calendar_event",
    "reschedule_calendar_event",
    "add_calendar_event_participants",
    "append_to_file",
    "create_file",
    "delete_file",
    "share_file",
)
BANKING_TOOLS = (
    "send_money",
    "schedule_transaction",
    "update_scheduled_transaction",
    "update_password",
    "update_user_info",
)


def _existing_coverage(catalog: dict) -> tuple[set[tuple[str, str]], int, int]:
    suite = get_suite("v1.2.2", "workspace")
    tool_schemas = {tool.name: tool.parameters for tool in suite.tools}
    linked, _ = _load_phase_records(
        list(DEFAULT_OBSERVED_EXPORT_ROOTS), catalog, tool_schemas
    )
    actionable_tasks = {
        row["id"] for row in catalog["user_tasks"] if row.get("possible_consequential_tools")
    }
    covered: set[tuple[str, str]] = set()
    fingerprints = set()
    for call, episode, _ in linked:
        fingerprints.add(_model_input_fingerprint(_observed_model_input(call)))
        if (
            episode.get("episode_kind") == "primary"
            and call.get("user_task_id") in actionable_tasks
            and call.get("injection_task_id")
        ):
            covered.add((call["user_task_id"], call["injection_task_id"]))
    return covered, len(linked), len(fingerprints)


def _select_pairs(catalog: dict, covered: set[tuple[str, str]], limit: int) -> tuple[list, list, dict]:
    task_rows = [
        row for row in catalog["user_tasks"] if row.get("possible_consequential_tools")
    ]
    injection_rows = catalog["injection_tasks"]
    injection_tools = {
        row["id"]: set(row.get("reference_tools", [])) for row in injection_rows
    }
    pairs = []
    for task in task_rows:
        task_id = task["id"]
        task_tools = set(task["possible_consequential_tools"])
        for injection in injection_rows:
            injection_id = injection["id"]
            pair = (task_id, injection_id)
            if pair in covered:
                continue
            overlap = task_tools & injection_tools[injection_id]
            pairs.append(
                {
                    "user_task_id": task_id,
                    "injection_task_id": injection_id,
                    "relevance_score": len(overlap),
                    "overlapping_reference_tools": sorted(overlap),
                }
            )
    if limit < 1 or limit > len(pairs):
        raise ValueError(f"limit must be between 1 and the available {len(pairs)} pairs")

    selected = []
    remaining = list(pairs)
    task_counts: Counter[str] = Counter()
    injection_counts: Counter[str] = Counter()
    while remaining and len(selected) < limit:
        remaining.sort(
            key=lambda pair: (
                -pair["relevance_score"],
                task_counts[pair["user_task_id"]],
                injection_counts[pair["injection_task_id"]],
                pair["user_task_id"],
                pair["injection_task_id"],
            )
        )
        pair = remaining.pop(0)
        selected.append(pair)
        task_counts[pair["user_task_id"]] += 1
        injection_counts[pair["injection_task_id"]] += 1
    return selected, remaining, {
        "selected_task_counts": dict(sorted(task_counts.items())),
        "selected_injection_counts": dict(sorted(injection_counts.items())),
        "relevance_overlap_pair_count": sum(
            item["relevance_score"] > 0 for item in selected
        ),
        "relevance_no_overlap_pair_count": sum(
            item["relevance_score"] == 0 for item in selected
        ),
    }


def _assign_attacks(pairs: list[dict]) -> dict[str, list[dict]]:
    by_attack = {attack: [] for attack in ATTACKS}
    for index, pair in enumerate(pairs):
        attack = ATTACKS[index % len(ATTACKS)]
        by_attack[attack].append(pair)
    return by_attack


def _make_batches(
    pairs_by_attack: dict[str, list[dict]],
    path_prefix: str,
    id_prefix: str,
) -> list[dict]:
    batches_by_attack: dict[str, list[dict]] = {}
    for attack, pairs in pairs_by_attack.items():
        batches = []
        for start in range(0, len(pairs), BATCH_SIZE):
            chunk = pairs[start : start + BATCH_SIZE]
            batch_number = start // BATCH_SIZE + 1
            suffix = "important" if attack == "important_instructions" else "tool-knowledge"
            batch_id = f"{id_prefix}-{suffix}-{batch_number:02d}"
            output_dir = f"runs/data_collection/{path_prefix}/{batch_id}"
            export_dir = f"data/exports/{path_prefix}/{batch_id}"
            batches.append(
                {
                    "id": batch_id,
                    "run_mode": "attacked_only",
                    "attack": attack,
                    "user_tasks": list(dict.fromkeys(p["user_task_id"] for p in chunk)),
                    "injection_tasks": list(
                        dict.fromkeys(p["injection_task_id"] for p in chunk)
                    ),
                    "attacked_task_pairs": [
                        [p["user_task_id"], p["injection_task_id"]] for p in chunk
                    ],
                    "run_attack_goal_checks": False,
                    "output_dir": output_dir,
                    "export_dir": export_dir,
                }
            )
        batches_by_attack[attack] = batches
    result = []
    for index in range(max(map(len, batches_by_attack.values()), default=0)):
        for attack in ATTACKS:
            if index < len(batches_by_attack[attack]):
                result.append(batches_by_attack[attack][index])
    return result


def _workspace_plan(catalog: dict, batches: list[dict], plan_id: str) -> dict:
    return {
        "plan_id": plan_id,
        "source_id": f"agentdojo-0.1.35-workspace-v1.2.2-{plan_id}",
        "benchmark_version": "v1.2.2",
        "suite": "workspace",
        "provider": "deepseek",
        "model": "deepseek-flash",
        "model_id": None,
        "temperature": 0.0,
        "decisions": ["ALLOW"],
        "repetitions": 1,
        "max_episode_attempts": 1,
        "consequential_tools": list(CONSEQUENTIAL_TOOLS),
        "tool_output_format": "yaml",
        "stop_policy": {
            "max_primary_episodes": sum(len(b["attacked_task_pairs"]) for b in batches),
            "max_unique_observed_calls": 250,
            "stop_at_batch_boundary_when_unique_observed_calls_reaches": 250,
            "no_retries": True,
            "no_repeat_task_injection_pair": True,
            "no_benign_reruns": True,
            "run_attack_goal_checks": False,
        },
        "batches": batches,
    }


def _banking_plan() -> dict:
    pairs = [
        ("user_task_3", "injection_task_0", "important_instructions"),
        ("user_task_9", "injection_task_4", "tool_knowledge"),
        ("user_task_14", "injection_task_7", "important_instructions"),
    ]
    batches = []
    for index, (task, injection, attack) in enumerate(pairs, start=1):
        batch_id = f"banking-compat-{index:02d}"
        batches.append(
            {
                "id": batch_id,
                "run_mode": "attacked_only",
                "attack": attack,
                "user_tasks": [task],
                "injection_tasks": [injection],
                "attacked_task_pairs": [[task, injection]],
                "run_attack_goal_checks": False,
                "output_dir": f"runs/data_collection/banking-compatibility/{batch_id}",
                "export_dir": f"data/exports/banking-compatibility/{batch_id}",
            }
        )
    return {
        "plan_id": "banking-compatibility-validation",
        "source_id": "agentdojo-0.1.35-banking-v1.2.2-compatibility-validation",
        "benchmark_version": "v1.2.2",
        "suite": "banking",
        "provider": "deepseek",
        "model": "deepseek-flash",
        "model_id": None,
        "temperature": 0.0,
        "decisions": ["ALLOW"],
        "repetitions": 1,
        "max_episode_attempts": 1,
        "consequential_tools": list(BANKING_TOOLS),
        "tool_output_format": "yaml",
        "stop_policy": {
            "max_primary_episodes": 3,
            "no_retries": True,
            "require_compatibility_review_before_any_larger_banking_run": True,
        },
        "batches": batches,
    }


def _token_budget(primary_episode_count: int, reserve_episode_count: int) -> dict:
    phase_root = Path("data/exports/real-traces-phase-1")
    primary_count = 0
    total_tokens = 0
    primary_calls = 0
    for path in phase_root.glob("*/collection_summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        primary_count += summary.get("primary_episode_count", 0)
        total_tokens += summary.get("token_usage", {}).get("total_tokens", 0)
        primary_calls += summary.get("candidate_consequential_call_count", 0)
    if not primary_count:
        raise ValueError("Phase-one summaries contain no primary episodes")
    mean_tokens = total_tokens / primary_count
    call_yield = primary_calls / primary_count
    reserve = {
        "episodes": reserve_episode_count,
        "expected_calls_at_phase1_yield": round(reserve_episode_count * call_yield, 1),
        "mean_token_estimate": round(reserve_episode_count * mean_tokens),
        "two_x_safety_estimate": round(reserve_episode_count * mean_tokens * 2),
    }
    return {
        "phase1_primary_episode_count": primary_count,
        "phase1_observed_consequential_call_count": primary_calls,
        "phase1_calls_per_primary_episode": round(call_yield, 3),
        "phase1_total_tokens_from_summaries": total_tokens,
        "phase1_mean_tokens_per_primary_episode": round(mean_tokens),
        "main": {
            "episodes": primary_episode_count,
            "expected_calls_at_phase1_yield": round(primary_episode_count * call_yield, 1),
            "mean_token_estimate": round(primary_episode_count * mean_tokens),
            "two_x_safety_estimate": round(primary_episode_count * mean_tokens * 2),
        },
        "workspace_reserve": reserve,
        "combined_max_episodes_if_reserve_is_approved": (
            primary_episode_count + reserve_episode_count
        ),
        "combined_two_x_safety_estimate_if_reserve_is_approved": (
            round((primary_episode_count + reserve_episode_count) * mean_tokens * 2)
        ),
        "note": (
            "Estimates use phase-one aggregate usage and are not a token hard cap. "
            "Provider calls can exceed the estimate; inspect batch summaries and stop "
            "at configured episode/candidate limits."
        ),
    }


def _dry_run_plan(plan_path: Path) -> dict:
    from scripts.collect_data import _plan_batches
    from tool_call_auditor.workspace_runner import validate_config

    plan, batches = _plan_batches(plan_path)
    reports = []
    for batch_id, config in batches:
        report = validate_config(config, require_model=False)
        report.update(
            {
                "plan_id": plan["plan_id"],
                "batch_id": batch_id,
                "source_id": config.source_id,
                "output_dir": config.output_dir.as_posix(),
                "export_dir": config.export_dir.as_posix(),
                "api_calls": 0,
                "fresh_output_paths": not config.output_dir.exists()
                and not config.export_dir.exists(),
            }
        )
        reports.append(report)
    primary_runs = sum(item["planned_total_agent_runs"] for item in reports)
    max_runs = sum(
        item["planned_total_agent_runs"] * item["max_episode_attempts"]
        for item in reports
    )
    return {
        "plan_id": plan["plan_id"],
        "totals": {
            "planned_primary_episodes": sum(
                item["planned_primary_episodes"] for item in reports
            ),
            "planned_attack_goal_checks": sum(
                item["planned_attack_goal_checks"] for item in reports
            ),
            "planned_total_agent_runs": primary_runs,
            "max_episode_attempts": max(
                (item["max_episode_attempts"] for item in reports), default=1
            ),
            "max_possible_agent_runs_with_retries": max_runs,
            "additional_retry_run_budget": max_runs - primary_runs,
            "unique_user_tasks": sorted(
                {task for item in reports for task in item["user_tasks"]}
            ),
            "attack_coverage": sorted(
                {
                    item["attack"]
                    for item in reports
                    if item["planned_attacked_episodes"]
                }
            ),
            "injection_task_coverage": sorted(
                {
                    pair["injection_task_id"]
                    for item in reports
                    for pair in item["attacked_task_pairs"]
                }
            ),
            "all_output_paths_fresh": all(
                item["fresh_output_paths"] for item in reports
            ),
        },
        "batches": reports,
    }


def prepare_plan(
    catalog_path: Path = Path("data/manifests/agentdojo_workspace_v1.2.2_catalog.json"),
    limit: int = MAIN_EPISODE_CAP,
) -> tuple[dict, dict, dict, dict]:
    catalog = _read_catalog(catalog_path)
    covered, observed_raw, observed_unique = _existing_coverage(catalog)
    selected, reserve_pairs, coverage = _select_pairs(catalog, covered, limit)
    selected_by_attack = _assign_attacks(selected)
    reserve_by_attack = _assign_attacks(reserve_pairs)
    main_batches = _make_batches(
        selected_by_attack, "real-traces-phase-2", "phase2"
    )
    reserve_batches = _make_batches(
        reserve_by_attack, "real-traces-phase-2-reserve", "phase2-reserve"
    )
    main_plan = _workspace_plan(catalog, main_batches, PLAN_ID)
    reserve_plan = _workspace_plan(
        catalog, reserve_batches, f"{PLAN_ID}-reserve"
    )
    reserve_plan["stop_policy"]["max_primary_episodes"] = len(reserve_pairs)
    reserve_plan["stop_policy"]["max_unique_observed_calls"] = 250
    bank_plan = _banking_plan()
    budget = _token_budget(len(selected), len(reserve_pairs))
    _, _, _, current_candidate_report = build_dry_run_report()
    current_constructed_count = current_candidate_report["new_constructed_count"]
    historical_duplicate_rate = (
        (observed_raw - observed_unique) / observed_raw if observed_raw else 0.0
    )
    projected_new_calls = budget["main"]["expected_calls_at_phase1_yield"]
    projected_new_unique_calls = round(
        projected_new_calls * (1 - historical_duplicate_rate)
    )
    current_candidate_rate = current_constructed_count / max(
        1, current_candidate_report["input"]["attacked_context_call_count"]
    )
    projected_main_constructed = round(projected_new_calls * current_candidate_rate)
    reserve_new_calls = budget["workspace_reserve"]["expected_calls_at_phase1_yield"]
    projected_reserve_unique_calls = round(
        reserve_new_calls * (1 - historical_duplicate_rate)
    )
    projected_reserve_constructed = round(reserve_new_calls * current_candidate_rate)
    projected_main_observed = observed_unique + projected_new_unique_calls
    projected_main_constructed_total = (
        current_constructed_count + projected_main_constructed
    )
    projected_full_observed = (
        projected_main_observed + projected_reserve_unique_calls
    )
    projected_full_constructed = (
        projected_main_constructed_total + projected_reserve_constructed
    )
    report = {
        "plan_id": PLAN_ID,
        "api_calls": 0,
        "source_type": "planning_only",
        "observed_exports": {
            "raw_call_count": observed_raw,
            "unique_model_input_count": observed_unique,
            "duplicate_rows_removed": observed_raw - observed_unique,
            "covered_task_injection_pair_count": len(covered),
        },
        "workspace": {
            "consequential_user_task_count": sum(
                bool(row.get("possible_consequential_tools"))
                for row in catalog["user_tasks"]
            ),
            "possible_pair_count": sum(
                bool(row.get("possible_consequential_tools"))
                for row in catalog["user_tasks"]
            )
            * len(catalog["injection_tasks"]),
            "unrun_unique_pairs_available": len(selected) + len(reserve_pairs),
            "main_episodes": len(selected),
            "reserve_episodes": len(reserve_pairs),
            "batch_count": len(main_batches),
            "reserve_batch_count": len(reserve_batches),
            "attack_counts": {
                attack: len(pairs) for attack, pairs in selected_by_attack.items()
            },
            "selected_coverage": coverage,
            "expected_new_observed_calls_at_phase1_yield": budget["main"][
                "expected_calls_at_phase1_yield"
            ],
            "expected_total_observed_calls_before_final_dedup": (
                observed_unique
                + budget["main"]["expected_calls_at_phase1_yield"]
            ),
            "stop_at_unique_observed_calls": 250,
        },
        "constructed_capacity": {
            "currently_context_supported_candidate_count": current_constructed_count,
            "current_near_duplicate_group_count": current_candidate_report[
                "near_duplicate_group_count"
            ],
            "current_constructed_methods": current_candidate_report[
                "construction_method_counts"
            ],
            "current_reviewable_candidate_count_after_cross_source_dedup": (
                observed_unique + current_constructed_count
            ),
            "current_candidate_row_gap_to_500": max(
                0, 500 - observed_unique - current_constructed_count
            ),
            "constructed_soft_target_gap_to_250": max(
                0, 250 - current_constructed_count
            ),
            "empirical_constructed_candidates_per_exposed_call": round(
                current_candidate_rate, 3
            ),
            "workspace_main_scenario": {
                "expected_new_observed_calls_before_dedup": projected_new_calls,
                "expected_new_unique_observed_calls_using_historical_duplicate_rate": (
                    projected_new_unique_calls
                ),
                "expected_new_constructed_candidates_using_current_yield": (
                    projected_main_constructed
                ),
                "expected_total_unique_observed_rows": projected_main_observed,
                "expected_total_constructed_rows": projected_main_constructed_total,
                "expected_total_exact_unique_candidate_rows": (
                    projected_main_observed + projected_main_constructed_total
                ),
                "expected_gap_to_500_rows": max(
                    0,
                    500 - projected_main_observed - projected_main_constructed_total,
                ),
            },
            "workspace_main_plus_reserve_scenario": {
                "expected_total_unique_observed_rows": projected_full_observed,
                "expected_total_constructed_rows": projected_full_constructed,
                "expected_total_exact_unique_candidate_rows": (
                    projected_full_observed + projected_full_constructed
                ),
                "expected_gap_to_500_rows": max(
                    0,
                    500 - projected_full_observed - projected_full_constructed,
                ),
            },
            "scenario_assumptions": {
                "phase_one_calls_per_primary_episode": (
                    budget["phase1_calls_per_primary_episode"]
                ),
                "historical_duplicate_fraction": round(historical_duplicate_rate, 3),
                "historical_duplicate_basis": (
                    f"{observed_raw - observed_unique} duplicate rows in {observed_raw} "
                    "observed call rows across legacy and phase-one exports"
                ),
                "future_candidate_yield_assumes_current_rate": current_candidate_rate,
            },
            "note": (
                "Current actual capacity is a dry-run result, not a projection. "
                "No future constructed yield is guaranteed by episode count."
            ),
        },
        "token_budget": budget,
        "banking": {
            "planned_validation_episodes": 3,
            "suite_count_in_scope": 1,
            "consequential_tools": list(BANKING_TOOLS),
            "plan_path": BANKING_PLAN.as_posix(),
            "validation_run_authorized": False,
            "compatibility_checks": [
                "get_suite(v1.2.2, banking) succeeds",
                "all five consequential tool schemas are present",
                "PilotConfig and validate_config accept a single attacked task pair without a model",
                "exactly three manual validation episodes; no retries or attack-goal checks",
            ],
        },
        "stopping": {
            "target_total_exact_unique_candidates": 500,
            "soft_mix": {"observed": 250, "constructed": 250},
            "stop_when_target_reached": True,
            "stop_after_main_and_reserve_episode_caps": True,
            "do_not_run_more_workspace_pairs_after_reserve": True,
            "if_short_after_workspace": "report gap; prepare banking expansion only after user approval",
        },
    }
    return main_plan, reserve_plan, bank_plan, report


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic no-API phase-two and banking plans."
    )
    parser.add_argument("--limit", type=int, default=MAIN_EPISODE_CAP)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    main_plan, reserve_plan, bank_plan, report = prepare_plan(limit=args.limit)
    report["paths"] = {
        "main_plan": PRIMARY_PLAN.as_posix(),
        "reserve_plan": RESERVE_PLAN.as_posix(),
        "banking_plan": BANKING_PLAN.as_posix(),
        "summary": SUMMARY_PATH.as_posix(),
        "dry_run": DRY_RUN_PATH.as_posix(),
        "banking_dry_run": BANKING_SUMMARY_PATH.as_posix(),
    }
    if args.write:
        paths = [
            PRIMARY_PLAN,
            RESERVE_PLAN,
            BANKING_PLAN,
            SUMMARY_PATH,
            DRY_RUN_PATH,
            BANKING_SUMMARY_PATH,
        ]
        existing = [path.as_posix() for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite plan artifacts: {', '.join(existing)}"
            )
        _write_new(
            PRIMARY_PLAN,
            yaml.safe_dump(main_plan, sort_keys=False, allow_unicode=True),
        )
        _write_new(
            RESERVE_PLAN,
            yaml.safe_dump(reserve_plan, sort_keys=False, allow_unicode=True),
        )
        _write_new(
            BANKING_PLAN,
            yaml.safe_dump(bank_plan, sort_keys=False, allow_unicode=True),
        )
        _write_new(
            SUMMARY_PATH,
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        main_dry_run = _dry_run_plan(PRIMARY_PLAN)
        reserve_dry_run = _dry_run_plan(RESERVE_PLAN)
        banking_dry_run = _dry_run_plan(BANKING_PLAN)
        _write_new(
            DRY_RUN_PATH,
            json.dumps(
                {
                    "main": main_dry_run,
                    "reserve": reserve_dry_run,
                    "api_calls": 0,
                    "candidate_tools_executed": 0,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
        )
        _write_new(
            BANKING_SUMMARY_PATH,
            json.dumps(
                {
                    "plan_id": banking_dry_run["plan_id"],
                    "dry_run": banking_dry_run,
                    "api_calls": 0,
                    "candidate_tools_executed": 0,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
        )
        report["dry_run_totals"] = {
            "main": main_dry_run["totals"],
            "reserve": reserve_dry_run["totals"],
            "banking": banking_dry_run["totals"],
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
