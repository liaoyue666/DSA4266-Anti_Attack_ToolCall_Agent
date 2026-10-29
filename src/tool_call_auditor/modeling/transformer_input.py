"""Tokenizer-aware input construction for a 512-token frozen encoder."""

from __future__ import annotations

import json
from typing import Any

from .serialization import _observation_text, select_observations

CRITICAL_ARGUMENTS = (
    "recipients",
    "participants",
    "email",
    "file_id",
    "attachments",
    "permission",
    "access_level",
    "owner",
    "path",
    "filename",
    "event_id",
    "calendar_id",
)


def serialize_action(action: dict[str, Any]) -> str:
    arguments = action.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    critical = {key: arguments[key] for key in CRITICAL_ARGUMENTS if key in arguments}
    remaining = {key: value for key, value in arguments.items() if key not in critical}
    return "\n".join(
        (
            f"tool_name={action.get('tool_name', '')}",
            "critical_arguments=" + json.dumps(critical, ensure_ascii=False, sort_keys=True),
            "remaining_arguments=" + json.dumps(remaining, ensure_ascii=False, sort_keys=True),
        )
    )


def _encode_with_head_tail(tokenizer, text: str, budget: int, *, tail_fraction: float = 0.25) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return ids
    tail = int(budget * tail_fraction)
    head = budget - tail
    return ids[:head] + ids[-tail:]


def encode_model_input(
    tokenizer,
    model_input: dict[str, Any],
    *,
    request_budget: int = 96,
    observation_budget: int = 256,
    action_budget: int = 156,
    max_observations: int = 3,
) -> dict[str, list[int]]:
    """Encode three fields independently so observations cannot erase actions."""

    request = "[USER REQUEST]\n" + str(model_input.get("original_user_request", ""))
    observations = model_input.get("observations_before_call", [])
    selected = select_observations(observations, max_observations=max_observations)
    observation = "[UNTRUSTED OBSERVATIONS]\n" + "\n---\n".join(
        _observation_text(item) for item in selected
    )
    action = "[PROPOSED ACTION]\n" + serialize_action(model_input.get("proposed_action", {}))

    request_ids = _encode_with_head_tail(tokenizer, request, request_budget, tail_fraction=0.15)
    observation_ids = _encode_with_head_tail(
        tokenizer, observation, observation_budget, tail_fraction=0.50
    )
    action_ids = _encode_with_head_tail(tokenizer, action, action_budget, tail_fraction=0.50)
    input_ids = [tokenizer.cls_token_id, *request_ids, *observation_ids, *action_ids, tokenizer.sep_token_id]
    if len(input_ids) > 512:
        raise ValueError(f"Encoded input has {len(input_ids)} tokens; budgets must total at most 510")
    attention_mask = [1] * len(input_ids)
    padding = 512 - len(input_ids)
    return {
        "input_ids": input_ids + [tokenizer.pad_token_id] * padding,
        "attention_mask": attention_mask + [0] * padding,
    }
