from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.package_dataset_delivery import (
    _annotation_row,
    _model_input_fingerprint,
    _observed_rows,
    _review_rows,
)


def _observed_call(
    call_id: str, task_group_id: str = "workspace:user_task_1"
) -> tuple[dict, dict, object]:
    call = {
        "call_id": call_id,
        "episode_id": f"episode-{call_id}",
        "source_id": f"source-{call_id}",
        "source_name": f"source-{call_id}",
        "_source_name": f"source-{call_id}",
        "_source_export_path": f"exports/{call_id}",
        "source_trace_path": f"traces/{call_id}.json",
        "task_group_id": task_group_id,
        "user_task_id": task_group_id.rsplit(":", 1)[-1],
        "injection_task_id": None,
        "original_user_request": "Send a message to Alex.",
        "tool_name": "send_email",
        "tool_arguments": {"recipients": ["alex@example.com"], "subject": "Hello"},
        "observation_provenance": [
            {
                "tool_name": "search_emails",
                "tool_arguments": {"query": "Alex"},
                "content": "Alex requested a meeting.",
                "error": None,
                "untrusted": False,
            }
        ],
    }
    episode = {"episode_id": call["episode_id"]}
    return call, episode, Path(call["source_trace_path"])


def test_observed_exact_duplicates_keep_every_parent_source() -> None:
    first = _observed_call("call-1")
    duplicate = _observed_call("call-2")
    row_list, summary = _observed_rows([first, duplicate])

    assert len(row_list) == 1
    assert summary == {
        "raw_call_count": 2,
        "unique_call_count": 1,
        "duplicate_rows_removed": 1,
        "duplicate_fingerprint_group_count": 1,
    }
    row = row_list[0]
    assert [ref["call_id"] for ref in row["parent_sources"]] == ["call-1", "call-2"]
    assert row["label"] is None
    assert row["model_input_fingerprint"] == _model_input_fingerprint(row["model_input"])


def test_annotation_model_input_excludes_review_metadata_and_is_json_serializable() -> None:
    model_input = {
        "original_user_request": "Send a message to Alex.",
        "observations_before_call": [],
        "proposed_action": {
            "tool_name": "send_email",
            "arguments": {"recipients": ["alex@example.com"]},
        },
    }
    constructed = {
        "candidate_id": "constructed-1",
        "source_type": "constructed",
        "parent": {"task_group_id": "workspace:user_task_1"},
        "parent_references": [
            {
                "call_id": "parent-call",
                "task_group_id": "workspace:user_task_1",
                "source_id": "source-1",
            }
        ],
        "model_input": model_input,
        "label": None,
        "near_duplicate_group_id": "near-group-1",
        "review_metadata": {
            "construction_intent": "expected_unauthorized",
            "review_reason": "Injected target is not authorized.",
        },
    }

    row = _annotation_row(constructed)
    encoded = json.dumps(row)
    decoded = json.loads(encoded)

    assert set(decoded["model_input"]) == {
        "original_user_request",
        "observations_before_call",
        "proposed_action",
    }
    assert "review_metadata" not in decoded
    assert "construction_intent" not in decoded["model_input"]
    assert decoded["label"] is None
    assert decoded["task_group_id"] == "workspace:user_task_1"


def test_review_list_only_includes_matching_constructed_candidate_ids() -> None:
    candidates = [{"candidate_id": "keep-1"}, {"candidate_id": "keep-2"}]
    reviews = [
        {"candidate_id": "keep-1", "label": None},
        {"candidate_id": "keep-2", "label": None},
        {"candidate_id": "capped-out", "label": None},
    ]
    result = _review_rows(candidates, reviews)

    assert [row["candidate_id"] for row in result] == ["keep-1", "keep-2"]
    assert all(row["label"] is None for row in result)


def test_review_list_rejects_missing_or_labeled_record() -> None:
    with pytest.raises(ValueError, match="Missing review record"):
        _review_rows([{"candidate_id": "missing"}], [])
    with pytest.raises(ValueError, match="label is not null"):
        _review_rows([{"candidate_id": "labeled"}], [{"candidate_id": "labeled", "label": 1}])
