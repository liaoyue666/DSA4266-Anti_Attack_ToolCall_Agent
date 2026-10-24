from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from .identifiers import episode_identifier

_ID_NAMESPACE = UUID("74a14e6a-4c34-4fd9-bf55-8119839727a1")


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                records.append(record)
    return records


def _manifest_for(path: Path, root: Path) -> dict:
    for parent in (path.parent, *path.parents):
        manifest_path = parent / "run_manifest.json"
        if manifest_path.is_file():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        if parent == root:
            break
    return {}


def _legacy_run_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return "legacy-" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]


def _condition_from_path(path: Path, root: Path) -> tuple[int | None, str | None]:
    relative_parts = path.relative_to(root).parts
    repetition = next(
        (
            int(part.removeprefix("repeat-"))
            for part in relative_parts
            if part.startswith("repeat-") and part.removeprefix("repeat-").isdigit()
        ),
        None,
    )
    decision = next(
        (part.upper() for part in relative_parts if part.lower() in {"allow", "block"}),
        None,
    )
    return repetition, decision


def _episode_record(
    trace_path: Path,
    root: Path,
    raw_context: dict,
    api_events: dict[str, list[dict]],
) -> dict:
    manifest = _manifest_for(trace_path, root)
    run_id = manifest.get("run_id") or _legacy_run_id(trace_path, root)
    path_repetition, path_decision = _condition_from_path(trace_path, root)
    repetition = path_repetition
    decision = path_decision
    if repetition is None:
        repetition = manifest.get("repetition")
    if decision is None:
        decision = manifest.get("decision")

    user_task_id = raw_context.get("user_task_id")
    injection_task_id = raw_context.get("injection_task_id")
    attack_type = raw_context.get("attack_type")
    is_attack_goal_check = bool(
        user_task_id
        and user_task_id.startswith("injection_task_")
        and injection_task_id is None
    )
    episode_id = episode_identifier(
        run_id,
        repetition,
        decision,
        raw_context.get("suite_name"),
        user_task_id,
        attack_type,
        injection_task_id,
    )
    error = raw_context.get("error")
    utility = raw_context.get("utility")
    security = raw_context.get("security")
    if error is None and utility is None and manifest.get("status") == "failed":
        error = manifest.get("error")
    if error:
        status = "failed"
    elif utility is None:
        status = "incomplete"
    else:
        status = "completed"

    events = api_events.get(episode_id, [])
    usage_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    summed_usage: dict[str, int | None] = {}
    for field in usage_fields:
        values = [(event.get("usage") or {}).get(field) for event in events]
        summed_usage[field] = (
            sum(values) if values and all(value is not None for value in values) else None
        )
    durations = [event.get("request_duration_seconds") for event in events]
    return {
        "run_id": run_id,
        "episode_id": episode_id,
        "repetition": repetition,
        "decision_condition": decision,
        "episode_kind": "attack_goal_check" if is_attack_goal_check else "primary",
        "scenario_type": (
            "attack_goal_check"
            if is_attack_goal_check
            else "attack"
            if injection_task_id and attack_type not in (None, "none")
            else "normal"
        ),
        "suite_name": raw_context.get("suite_name"),
        "task_id": user_task_id,
        "user_task_id": user_task_id,
        "injection_task_id": injection_task_id,
        "injection_id": injection_task_id or (user_task_id if is_attack_goal_check else None),
        "attack_type": attack_type,
        "episode_status": status,
        "task_result": utility,
        "agentdojo_security_result": security,
        "attack_succeeded": (
            security
            if not is_attack_goal_check
            and injection_task_id
            and attack_type not in (None, "none")
            and isinstance(security, bool)
            else None
        ),
        "duration_seconds": raw_context.get("duration"),
        "error": error,
        "api_usage": {
            **summed_usage,
            "cost": None,
            "api_call_count": len(events) if events else None,
            "request_duration_seconds": (
                sum(durations)
                if durations and all(isinstance(value, (int, float)) for value in durations)
                else None
            ),
        },
        "trace_path": os.path.relpath(trace_path, root),
        "agentdojo_context": raw_context,
    }


def _collect_api_events(root: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for path in root.rglob("api_requests.jsonl"):
        for event in _read_jsonl(path):
            episode_id = event.get("episode_id")
            if episode_id:
                grouped.setdefault(episode_id, []).append(event)
    return grouped


def _collect_call_records(root: Path) -> list[dict]:
    merged: dict[str, dict] = {}
    for path in root.rglob("tool_call_decisions.jsonl"):
        manifest = _manifest_for(path, root)
        fallback_run_id = manifest.get("run_id") or _legacy_run_id(path, root)
        for index, event in enumerate(_read_jsonl(path)):
            call_id = event.get("call_id")
            if not call_id:
                call_id = str(uuid5(_ID_NAMESPACE, f"{path}:{index}"))
                event = {**event, "call_id": call_id, "run_id": fallback_run_id}
            if "episode_kind" not in event:
                event["episode_kind"] = (
                    "attack_goal_check"
                    if str(event.get("user_task_id", "")).startswith("injection_task_")
                    else "primary"
                )
            existing = merged.get(call_id, {})
            merged[call_id] = {**existing, **event}

    return [
        {**record, "label": None}
        for record in merged.values()
        if record.get("consequential") is True
        and record.get("episode_kind", "primary") == "primary"
    ]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary_path.replace(path)


def export_dataset(
    source_dir: Path,
    export_dir: Path,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    source_dir = source_dir.resolve()
    export_dir = export_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source run directory does not exist: {source_dir}")
    if export_dir.exists() and any(export_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Export directory {export_dir} already contains data; "
            "choose another export_dir or pass --overwrite."
        )

    api_events = _collect_api_events(source_dir)
    episodes = []
    for trace_path in sorted(source_dir.rglob("*.json")):
        if trace_path.name in {"summary.json", "run_manifest.json"}:
            continue
        raw_context = json.loads(trace_path.read_text(encoding="utf-8"))
        if not isinstance(raw_context, dict) or "messages" not in raw_context:
            continue
        episodes.append(_episode_record(trace_path, source_dir, raw_context, api_events))

    raw_calls = _collect_call_records(source_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    calls_path = export_dir / "raw_calls.jsonl"
    episodes_path = export_dir / "episodes.jsonl"
    _write_jsonl(calls_path, raw_calls)
    _write_jsonl(episodes_path, episodes)
    return calls_path, episodes_path
