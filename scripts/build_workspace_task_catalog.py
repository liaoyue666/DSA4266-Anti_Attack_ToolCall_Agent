from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from agentdojo.task_suite import get_suite


CONSEQUENTIAL_TOOLS = {
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
}
BENCHMARK_VERSION = "v1.2.2"
SUITE_NAME = "workspace"


def _reference_calls(task, environment) -> list[dict]:
    return [
        {"tool": call.function, "arguments": call.args}
        for call in task.ground_truth(environment)
    ]


def build_catalog() -> dict:
    suite = get_suite(BENCHMARK_VERSION, SUITE_NAME)
    environment = suite.load_and_inject_default_environment({})
    users = []
    for task_id, task in suite.user_tasks.items():
        reference_calls = _reference_calls(task, environment)
        reference_tools = sorted({call["tool"] for call in reference_calls})
        users.append(
            {
                "id": task_id,
                "difficulty": str(getattr(task, "DIFFICULTY", "")),
                "instruction": task.PROMPT,
                "reference_output": getattr(task, "GROUND_TRUTH_OUTPUT", None),
                "reference_calls": reference_calls,
                "reference_tools": reference_tools,
                "possible_consequential_tools": sorted(
                    set(reference_tools) & CONSEQUENTIAL_TOOLS
                ),
            }
        )

    injections = []
    for task_id, task in suite.injection_tasks.items():
        reference_calls = _reference_calls(task, environment)
        injections.append(
            {
                "id": task_id,
                "difficulty": str(getattr(task, "DIFFICULTY", "")),
                "goal": task.GOAL,
                "reference_calls": reference_calls,
                "reference_tools": sorted({call["tool"] for call in reference_calls}),
                "reference_behavior_note": (
                    "AgentDojo reference behavior only; not an authorization label."
                ),
            }
        )

    tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "consequential_in_this_project": tool.name in CONSEQUENTIAL_TOOLS,
        }
        for tool in suite.tools
    ]
    return {
        "agentdojo_package_version": importlib.metadata.version("agentdojo"),
        "benchmark_version": BENCHMARK_VERSION,
        "suite": SUITE_NAME,
        "user_task_count": len(users),
        "injection_task_count": len(injections),
        "tool_count": len(tools),
        "user_tasks": users,
        "injection_tasks": injections,
        "tools": tools,
        "authorization_note": (
            "Ground-truth/reference behavior describes benchmark utility or attack goals. "
            "It is not a call-level authorization label."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate installed AgentDojo workspace tasks and tools offline."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/agentdojo_workspace_v1.2.2_catalog.json"),
    )
    args = parser.parse_args()
    catalog = build_catalog()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "agentdojo_package_version": catalog["agentdojo_package_version"],
                "benchmark_version": catalog["benchmark_version"],
                "user_task_count": catalog["user_task_count"],
                "injection_task_count": catalog["injection_task_count"],
                "tool_count": catalog["tool_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
