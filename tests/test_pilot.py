import json
from pathlib import Path

import pytest

from tool_call_auditor.demo import run_episode


@pytest.mark.parametrize(
    ("kind", "decision", "expected_recipient"),
    [
        ("benign", "ALLOW", "alice@example.com"),
        ("attacked", "ALLOW", "attacker@example.com"),
        ("benign", "BLOCK", None),
        ("attacked", "BLOCK", None),
    ],
)
def test_fixed_decision_gate(
    tmp_path: Path, kind: str, decision: str, expected_recipient: str | None
) -> None:
    log_path = tmp_path / "decisions.jsonl"
    outcome = run_episode(kind, decision, log_path)

    if expected_recipient is None:
        assert outcome["sent_emails"] == []
        assert "BLOCKED_BY_AUTHORIZATION_GATE" in outcome["final_message"]
    else:
        assert outcome["sent_emails"][0]["recipient"] == expected_recipient

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["original_user_request"]
    assert record["untrusted_observation"]
    assert record["tool_name"] == "send_email"
    assert record["decision"] == decision
    assert record["intercepted_before_execution"] is True
