from __future__ import annotations

from typing import Any


def episode_identifier(
    run_id: str,
    repetition: int | None,
    decision: str | None,
    suite_name: str | None,
    user_task_id: str | None,
    attack_type: str | None,
    injection_task_id: str | None,
) -> str:
    is_auxiliary_check = bool(user_task_id and user_task_id.startswith("injection_task_"))
    scenario = (
        "attack_goal_check"
        if is_auxiliary_check
        else "attack"
        if injection_task_id and attack_type not in (None, "none")
        else "normal"
    )
    task_id = user_task_id or "unknown-task"
    injection_id = injection_task_id or (
        task_id if is_auxiliary_check else "none"
    )
    parts: tuple[Any, ...] = (
        run_id,
        f"repeat-{repetition if repetition is not None else 'unknown'}",
        decision or "unknown-decision",
        suite_name or "unknown-suite",
        scenario,
        task_id,
        attack_type or "none",
        injection_id,
    )
    return "/".join(str(part) for part in parts)
