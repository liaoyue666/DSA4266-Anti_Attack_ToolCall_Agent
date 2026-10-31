"""Independently verify provenance, split isolation and strict pre-call inputs."""

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--prior-batch", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("builder", Path(__file__).with_name("prepare_public_trace_batch_400.py"))
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    root = args.batch
    rows = [json.loads(s) for s in (root / "annotation_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    prior = [json.loads(s) for s in (args.project / "data/annotations/dataset_502_v1/labeled_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    old120 = [json.loads(s) for s in (args.prior_batch / "annotation_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    old120_sources = {p["upstream_path"] for r in old120 for p in r["parent_sources"]}
    prior += old120
    tree = subprocess.check_output(["git", "ls-tree", "-r", builder.COMMIT, "runs"], cwd=args.upstream, text=True)
    blobs = {line.split("\t", 1)[1]: line.split("\t", 1)[0].split()[2] for line in tree.splitlines()}
    verified_sources = set()
    old_fingerprints = {builder.fingerprint(r["model_input"]) for r in prior}
    with (args.project / "data/splits/dataset_502_v1/task_group_assignments.csv").open(encoding="utf-8") as f:
        assignments = {r["task_group_id"]: r for r in csv.DictReader(f)}
    assert len(rows) == len({r["candidate_id"] for r in rows}) == 400
    for row in rows:
        assert row["label"] is None
        assert assignments[row["task_group_id"]]["split"] == row["existing_split"] == "train"
        assert builder.fingerprint(row["model_input"]) == row["model_input_fingerprint"]
        assert row["model_input_fingerprint"] not in old_fingerprints
        assert set(row["model_input"]) == {"original_user_request", "observations_before_call", "proposed_action"}
        assert row["model_input"]["proposed_action"]["tool_name"] in builder.TOOLS
        for parent in row["parent_sources"]:
            source = root / "sources" / parent["upstream_path"]
            body = source.read_bytes()
            assert hashlib.sha256(body).hexdigest() == parent["source_sha256"]
            assert parent["upstream_path"] not in old120_sources
            git_blob = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
            assert git_blob == blobs[parent["upstream_path"]]
            verified_sources.add(parent["upstream_path"])
            trace = json.loads(body)
            # Reconstruct from the exact source position, not just the ID/hash.
            found = [m for m, p in builder.extract(trace, {})
                     if p["message_index"] == parent["message_index"] and p["call_index"] == parent["call_index"]]
            assert found == [row["model_input"]]
    review = [json.loads(s) for s in (root / "review_metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["candidate_id"] for r in review} == {r["candidate_id"] for r in rows}
    assert all(r["label"] is None and r["review_status"] == "pending" and not r["reviewer"] for r in review)
    with (root / "review_sheet.csv").open(encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))
    by_id = {r["candidate_id"]: r for r in rows}
    assert len(sheet) == 400
    for r in sheet:
        full = json.loads((root / r["full_context_file"]).read_text(encoding="utf-8"))
        assert full["candidate_id"] == r["candidate_id"]
        assert full["model_input"] == by_id[r["candidate_id"]]["model_input"]
        assert full["label"] is None and not r["label"]
    stats = json.loads((root / "dataset_statistics.json").read_text())
    ra = lambda r: builder.digest([r["model_input"]["original_user_request"],r["model_input"]["proposed_action"]])
    old_ra = {ra(r) for r in prior}
    assert len({ra(r) for r in rows}) == stats["within_batch_unique_request_actions"]
    assert sum(ra(r) in old_ra for r in rows) == stats["prior_622_request_action_overlap_rows"]
    manifest = json.loads((root / "source_manifest.json").read_text())
    assert len(manifest) == len(verified_sources) == stats["source_traces_included"]
    for entry in manifest:
        assert hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    # Regression check: the proposed action's own result cannot appear in input.
    trace = {"messages": [
        {"role": "user", "content": "Read the file."},
        {"role": "assistant", "content": "Do not export this thought", "tool_calls": [{"id": "read", "function": "read_file", "args": {}}]},
        {"role": "tool", "tool_call_id": "read", "content": "PRIOR_OBSERVATION", "error": None},
        {"role": "assistant", "tool_calls": [{"id": "send", "function": "send_email", "args": {"recipients": ["other@example.org"]}}]},
        {"role": "tool", "tool_call_id": "send", "content": "FUTURE_RESULT", "error": None},
    ]}
    extracted = list(builder.extract(trace, {}))[0][0]
    assert "PRIOR_OBSERVATION" in json.dumps(extracted)
    assert "FUTURE_RESULT" not in json.dumps(extracted)
    assert "Do not export this thought" not in json.dumps(extracted)
    report = {"status": "PASS", "candidates": len(rows), "unique": True,
              "all_labels_null_and_pending": True, "existing_502_exact_overlap": 0,
              "old_validation_test_task_overlap": 0,
              "prior_120_exact_overlap": 0, "prior_120_source_episode_overlap": 0,
              "source_git_blobs_match_pinned_commit": len(verified_sources),
              "within_batch_unique_request_actions": len({ra(r) for r in rows}),
              "prior_622_request_action_overlap_rows": sum(ra(r) in old_ra for r in rows),
              "every_parent_reconstructed_from_source": True,
              "pre_call_boundary_regression": "PASS", "review_case_alignment": "PASS"}
    (root / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
