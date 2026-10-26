from pathlib import Path

from agentdojo.task_suite import get_suite

from scripts.prepare_phase2_plan import (
    BANKING_TOOLS,
    _banking_plan,
    _make_batches,
    _select_pairs,
)
from scripts.check_candidate_progress import _next_action
from tool_call_auditor.config import PilotConfig
from tool_call_auditor.workspace_runner import validate_config


def test_pair_selection_excludes_covered_pairs_and_balances_tasks() -> None:
    catalog = {
        "user_tasks": [
            {"id": "user_task_1", "possible_consequential_tools": ["send_email"]},
            {"id": "user_task_2", "possible_consequential_tools": ["send_email"]},
        ],
        "injection_tasks": [
            {"id": "injection_task_0", "reference_tools": ["send_email"]},
            {"id": "injection_task_1", "reference_tools": ["delete_file"]},
            {"id": "injection_task_2", "reference_tools": ["send_email"]},
        ],
    }
    selected, reserve, summary = _select_pairs(
        catalog, {("user_task_1", "injection_task_0")}, limit=4
    )

    selected_pairs = {
        (pair["user_task_id"], pair["injection_task_id"]) for pair in selected
    }
    assert len(selected) == 4
    assert len(selected_pairs) == 4
    assert ("user_task_1", "injection_task_0") not in selected_pairs
    assert len(reserve) == 1
    assert summary["relevance_overlap_pair_count"] >= 2
    assert max(summary["selected_task_counts"].values()) <= 2


def test_batches_keep_attack_and_output_paths_separate() -> None:
    pairs = [
        {"user_task_id": f"user_task_{index}", "injection_task_id": "injection_task_0"}
        for index in range(12)
    ]
    batches = _make_batches(
        {"important_instructions": pairs, "tool_knowledge": []},
        "real-traces-phase-2",
        "phase2",
    )

    assert [len(batch["attacked_task_pairs"]) for batch in batches] == [10, 2]
    assert len({batch["output_dir"] for batch in batches}) == 2
    assert len({batch["export_dir"] for batch in batches}) == 2
    assert all(batch["run_mode"] == "attacked_only" for batch in batches)
    assert all(not batch["run_attack_goal_checks"] for batch in batches)


def test_banking_suite_schemas_and_config_preflight_are_compatible() -> None:
    suite = get_suite("v1.2.2", "banking")
    schemas = {tool.name: tool.parameters for tool in suite.tools}
    assert set(BANKING_TOOLS) <= schemas.keys()
    assert set(schemas["send_money"].model_fields) == {
        "recipient",
        "amount",
        "subject",
        "date",
    }
    assert set(schemas["schedule_transaction"].model_fields) == {
        "recipient",
        "amount",
        "subject",
        "date",
        "recurring",
    }
    assert "id" in schemas["update_scheduled_transaction"].model_fields
    assert "password" in schemas["update_password"].model_fields
    assert set(schemas["update_user_info"].model_fields) == {
        "first_name",
        "last_name",
        "street",
        "city",
    }

    plan = _banking_plan()
    batch = plan["batches"][0]
    config = PilotConfig.model_validate(
        {
            **{key: value for key, value in plan.items() if key not in {"plan_id", "batches", "stop_policy"}},
            **{key: value for key, value in batch.items() if key != "id"},
        }
    )
    preflight = validate_config(config, require_model=False)
    assert preflight["suite"] == "banking"
    assert preflight["model"] == "deepseek-flash"
    assert preflight["decisions"] == ["ALLOW"]
    assert preflight["planned_primary_episodes"] == 1
    assert not Path(config.output_dir).exists()


def test_progress_stop_policy_is_exact_and_bounded() -> None:
    assert _next_action(500, 250, 250, 500, 250, 250, "next", False, None, False) == (
        "STOP_TARGET_REACHED",
        None,
    )
    assert _next_action(400, 250, 150, 500, 250, 250, None, True, "reserve-01", False) == (
        "STOP_WORKSPACE_CAP_REACHED_REPORT_GAP",
        None,
    )
    assert _next_action(300, 220, 80, 500, 250, 250, None, True, "reserve-01", False) == (
        "RUN_NEXT_WORKSPACE_RESERVE_BATCH",
        "reserve-01",
    )
    assert _next_action(300, 200, 100, 500, 250, 250, "main-02", False, None, False) == (
        "RUN_NEXT_MAIN_BATCH",
        "main-02",
    )
