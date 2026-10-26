from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_constructed_candidates import (
    DEFAULT_OBSERVED_EXPORT_ROOTS,
    build_dry_run_report,
)

PHASE2_PLAN = Path("configs/real_trace_expansion_phase2.yaml")
RESERVE_PLAN = Path("configs/real_trace_expansion_phase2_reserve.yaml")
DEFAULT_TARGET = 500
OBSERVED_TARGET = 250
CONSTRUCTED_TARGET = 250


def _next_batch(plan_path: Path) -> tuple[str | None, bool]:
    if not plan_path.is_file():
        return None, False
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    for batch in plan.get("batches", []):
        summary_path = Path(batch["export_dir"]) / "collection_summary.json"
        if not summary_path.is_file():
            return batch["id"], False
    return None, True


def _next_action(
    total_unique: int,
    observed_unique: int,
    constructed_unique: int,
    target: int,
    observed_target: int,
    constructed_target: int,
    main_next: str | None,
    main_complete: bool,
    reserve_next: str | None,
    reserve_complete: bool,
) -> tuple[str, str | None]:
    if total_unique >= target:
        return "STOP_TARGET_REACHED", None
    if observed_unique >= observed_target and constructed_unique >= constructed_target:
        return "STOP_SOFT_MIX_REACHED", None
    if not main_complete:
        return "RUN_NEXT_MAIN_BATCH", main_next
    if observed_unique < observed_target and not reserve_complete:
        return "RUN_NEXT_WORKSPACE_RESERVE_BATCH", reserve_next
    return "STOP_WORKSPACE_CAP_REACHED_REPORT_GAP", None


def summarize_progress(
    target: int = DEFAULT_TARGET,
    observed_target: int = OBSERVED_TARGET,
    constructed_target: int = CONSTRUCTED_TARGET,
) -> dict:
    if min(target, observed_target, constructed_target) <= 0:
        raise ValueError("Candidate targets must be positive integers")
    possible_roots = list(DEFAULT_OBSERVED_EXPORT_ROOTS) + [
        Path("data/exports/real-traces-phase-2"),
        Path("data/exports/real-traces-phase-2-reserve"),
        Path("data/exports/banking-compatibility"),
    ]
    roots = [path for path in possible_roots if path.is_dir()]
    candidates, _, _, candidate_report = build_dry_run_report(export_roots=roots)
    observed_unique = candidate_report["observed_unique_model_input_count"]
    constructed_unique = len(candidates)
    total_unique = observed_unique + constructed_unique
    main_next, main_complete = _next_batch(PHASE2_PLAN)
    reserve_next, reserve_complete = _next_batch(RESERVE_PLAN)
    decision, next_batch = _next_action(
        total_unique,
        observed_unique,
        constructed_unique,
        target,
        observed_target,
        constructed_target,
        main_next,
        main_complete,
        reserve_next,
        reserve_complete,
    )

    observed_raw = candidate_report["observed_raw_call_count"]
    return {
        "target_exact_unique_candidate_rows": target,
        "observed_target_rows": observed_target,
        "constructed_target_rows": constructed_target,
        "observed_raw_rows": observed_raw,
        "observed_duplicate_rows_removed": observed_raw - observed_unique,
        "observed_unique_rows": observed_unique,
        "constructed_unique_rows_not_matching_observed": constructed_unique,
        "total_exact_unique_candidate_rows": total_unique,
        "gap_to_target": max(0, target - total_unique),
        "constructed_near_duplicate_groups": candidate_report[
            "constructed_near_duplicate_group_count"
        ],
        "near_duplicate_groups_are_not_row_counts": True,
        "label_assignment_performed": False,
        "api_calls": 0,
        "candidate_tools_executed": 0,
        "next_action": decision,
        "next_batch_id": next_batch,
        "workspace_main_plan_complete": main_complete,
        "workspace_reserve_plan_complete": reserve_complete,
        "workspace_plan_max_episode_count": 279,
        "exact_fingerprint_fields": [
            "original_user_request",
            "ordered_observations_before_call",
            "tool_name",
            "arguments",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report exact cross-source candidate counts and a bounded next step."
    )
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--observed-target", type=int, default=OBSERVED_TARGET)
    parser.add_argument("--constructed-target", type=int, default=CONSTRUCTED_TARGET)
    args = parser.parse_args()
    print(
        json.dumps(
            summarize_progress(
                args.target, args.observed_target, args.constructed_target
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
