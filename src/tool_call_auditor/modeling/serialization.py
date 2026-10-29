"""Deterministic, field-aware serialization of verifier inputs."""

from __future__ import annotations

import json
from typing import Any

INJECTION_MARKERS = (
    "<information>",
    "ignore previous",
    "ignore all previous",
    "before you can solve",
    "system message",
    "prompt injection",
)


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _observation_text(observation: dict[str, Any]) -> str:
    return "\n".join(
        (
            f"tool={observation.get('tool_name', '')}",
            f"arguments={_compact_json(observation.get('tool_arguments', {}))}",
            f"content={observation.get('content', '')}",
            f"error={observation.get('error', '') or ''}",
        )
    )


def select_observations(
    observations: list[dict[str, Any]],
    *,
    max_observations: int = 3,
) -> list[dict[str, Any]]:
    """Keep recent context plus the most recent observation with injection cues.

    The selection is label-blind. It never reads annotation metadata, source type,
    or construction metadata.
    """

    if max_observations < 1:
        raise ValueError("max_observations must be positive")
    if len(observations) <= max_observations:
        return observations

    selected_indices = set(range(len(observations) - max_observations, len(observations)))
    injection_indices = [
        index
        for index, observation in enumerate(observations)
        if any(marker in _observation_text(observation).lower() for marker in INJECTION_MARKERS)
    ]
    if injection_indices:
        selected_indices.add(injection_indices[-1])

    ordered = sorted(selected_indices)
    if len(ordered) > max_observations:
        required = injection_indices[-1] if injection_indices else ordered[-1]
        other = [index for index in reversed(ordered) if index != required]
        ordered = sorted([required, *other[: max_observations - 1]])
    return [observations[index] for index in ordered]


def _clip_middle(text: str, limit: int) -> str:
    if limit < 32:
        raise ValueError("text limit must be at least 32 characters")
    if len(text) <= limit:
        return text
    marker = "\n...[TRUNCATED]...\n"
    remaining = limit - len(marker)
    head = remaining // 2
    return text[:head] + marker + text[-(remaining - head) :]


def serialize_model_input(
    model_input: dict[str, Any],
    *,
    max_observations: int = 3,
    max_observation_chars: int = 8_000,
) -> str:
    """Serialize without ever truncating the proposed action.

    Character clipping is only a preprocessing guard for sparse baselines. The
    Transformer path applies tokenizer-aware per-field budgets separately.
    """

    observations = model_input.get("observations_before_call", [])
    if not isinstance(observations, list):
        raise ValueError("observations_before_call must be a list")
    selected = select_observations(observations, max_observations=max_observations)
    observation_text = "\n\n--- observation ---\n".join(
        _observation_text(observation) for observation in selected
    )
    observation_text = _clip_middle(observation_text, max_observation_chars)

    request = str(model_input.get("original_user_request", ""))
    action = model_input.get("proposed_action", {})
    return (
        "[USER REQUEST]\n"
        f"{request}\n\n"
        "[UNTRUSTED OBSERVATIONS]\n"
        f"{observation_text}\n\n"
        "[PROPOSED ACTION]\n"
        f"{_compact_json(action)}"
    )
