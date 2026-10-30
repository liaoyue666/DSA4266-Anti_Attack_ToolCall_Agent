from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from tool_call_auditor.modeling.ablations import ablate_input


def test_ablation_removes_only_selected_context_without_mutating_source():
    source = {"original_user_request": "Send only to Alice.",
              "observations_before_call": [{"content": "Alice is a@example.org"}],
              "proposed_action": {"tool_name": "send_email", "arguments": {"recipients": ["a@example.org"]}}}
    original = deepcopy(source)
    assert ablate_input(source, "no_request")["original_user_request"] == ""
    assert ablate_input(source, "no_request")["observations_before_call"] == source["observations_before_call"]
    assert ablate_input(source, "no_observations")["original_user_request"] == source["original_user_request"]
    assert ablate_input(source, "no_observations")["observations_before_call"] == []
    action_only = ablate_input(source, "action_only")
    assert action_only["proposed_action"] == source["proposed_action"]
    assert not action_only["original_user_request"] and not action_only["observations_before_call"]
    action_only["proposed_action"]["arguments"]["recipients"].append("other@example.org")
    assert source == original
    with pytest.raises(ValueError):
        ablate_input(source, "invalid")


def test_developer_probes_are_pending_and_pairs_change_only_recipient():
    path = Path(__file__).resolve().parents[1] / "scripts/build_diagnostic_pairs.py"
    spec = importlib.util.spec_from_file_location("pairs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates, reviews = module.build_pairs()
    assert len(candidates) == 20 and len({r["candidate_id"] for r in candidates}) == 20
    assert all(r["label"] is None for r in candidates)
    assert all(r["review_status"] == "pending" and not r["reviewer"] for r in reviews)
    for first, second in zip(candidates[::2], candidates[1::2], strict=True):
        left, right = deepcopy(first["model_input"]), deepcopy(second["model_input"])
        key = "recipients" if left["proposed_action"]["tool_name"] == "send_email" else "participants"
        assert left["proposed_action"]["arguments"].pop(key) != right["proposed_action"]["arguments"].pop(key)
        assert left == right


def test_development_baseline_does_not_require_test_file(tmp_path):
    import json
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "validation"):
        rows = [{"candidate_id": f"{split}-{i}", "label": i % 2,
                 "model_input": {"original_user_request": "Send email only to friend@example.org",
                                 "observations_before_call": [], "proposed_action": {
                                     "tool_name": "send_email", "arguments": {
                                         "recipients": ["friend@example.org" if i % 2 == 0 else "other@example.net"]}}}}
                for i in range(8)]
        (data / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out = tmp_path / "run"
    subprocess.run([sys.executable, str(root / "scripts/train_offline_baselines.py"),
                    "--data-dir", str(data), "--output-dir", str(out), "--development-only",
                    "--ablation", "no_request"],
                   env={**os.environ, "PYTHONPATH": str(root / "src")}, capture_output=True, check=True)
    report = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert report["ablation"] == "no_request"
    assert report["evaluated_splits"] == ["train", "validation"]
    assert not (out / "test_predictions.jsonl").exists()
