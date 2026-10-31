"""Collect pinned public AgentDojo traces; extract unlabeled review candidates."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import unicodedata
from urllib.request import Request, urlopen

COMMIT = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
REPO_URL = "https://github.com/ethz-spylab/agentdojo"
TOOLS = {"send_email", "append_to_file", "create_file", "share_file", "delete_file",
         "create_calendar_event", "reschedule_calendar_event", "add_calendar_event_participants"}
MODELS = {"gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18", "claude-3-5-sonnet-20240620",
          "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.0-flash-001",
          "gemini-1.5-pro-002", "gpt-4-turbo-2024-04-09", "claude-3-haiku-20240307",
          "claude-3-opus-20240229", "gemini-1.5-flash-002", "command-r-plus", "gpt-3.5-turbo-0125"}


def norm(v):
    if isinstance(v, str):
        return unicodedata.normalize("NFC", v)
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {norm(str(k)): norm(x) for k, x in v.items()}
    return v


def digest(v):
    return hashlib.sha256(json.dumps(norm(v), ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def fingerprint(m):
    return digest({"original_user_request": m["original_user_request"],
                   "observations_before_call": m["observations_before_call"],
                   "tool_name": m["proposed_action"]["tool_name"],
                   "arguments": m["proposed_action"]["arguments"]})


def text_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(x.get("text", "") for x in content if isinstance(x, dict))
    raise ValueError("Unsupported message content")


def action(call):
    function = call.get("function")
    if isinstance(function, dict):
        name, args = function["name"], function.get("arguments", {})
    else:
        name, args = function, call.get("args", {})
    if isinstance(args, str):
        args = json.loads(args)
    if not isinstance(name, str) or not isinstance(args, dict):
        raise ValueError("Invalid tool proposal")
    return name, args


def extract(trace, source):
    observations, calls = [], {}
    request = None
    for index, message in enumerate(trace["messages"]):
        if message["role"] == "user" and request is None:
            request = text_content(message["content"])
        elif message["role"] == "assistant":
            for ordinal, call in enumerate(message.get("tool_calls") or []):
                name, args = action(call)
                calls[call.get("id")] = (name, args)
                if name in TOOLS:
                    if request is None:
                        raise ValueError("Proposal precedes request")
                    m = {"original_user_request": request,
                         "observations_before_call": deepcopy(observations),
                         "proposed_action": {"tool_name": name, "arguments": args}}
                    yield m, {**source, "message_index": index, "call_index": ordinal,
                              "provider_call_id": call.get("id")}
        elif message["role"] == "tool":
            producing = message.get("tool_call")
            name, args = action(producing) if producing else calls.get(message.get("tool_call_id"), (None, None))
            observations.append({"tool_name": name, "tool_arguments": args,
                                 "content": text_content(message.get("content")),
                                 "error": message.get("error"), "untrusted": True})


def round_robin(rows, count):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task_group_id"], row["model_input"]["proposed_action"]["tool_name"])].append(row)
    for values in groups.values():
        values.sort(key=lambda r: digest({"seed": 4266, "id": r["candidate_id"]}))
    result = []
    while groups and len(result) < count:
        for key in sorted(list(groups)):
            result.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
            if len(result) == count:
                break
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--prior-batch", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=400)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be empty")
    if not 1 <= args.max_candidates <= 1000:
        raise ValueError("Candidate count out of range")
    with (args.project / "data/splits/dataset_502_v1/task_group_assignments.csv").open(encoding="utf-8") as f:
        assignments = {r["task_group_id"]: r for r in csv.DictReader(f)}
    train_tasks = {g.split(":", 1)[1] for g, r in assignments.items() if r["split"] == "train"}
    existing_path = args.project / "data/annotations/dataset_502_v1/labeled_calls.jsonl"
    existing = [json.loads(s) for s in existing_path.read_text(encoding="utf-8").splitlines()]
    prior_path = args.prior_batch / "annotation_candidates.jsonl"
    prior = [json.loads(s) for s in prior_path.read_text(encoding="utf-8").splitlines()]
    blocked = {fingerprint(r["model_input"]) for r in existing + prior}
    prior_source_paths = {p["upstream_path"] for r in prior for p in r["parent_sources"]}
    existing_requests = defaultdict(set)
    for r in existing:
        existing_requests[r["task_group_id"]].add(" ".join(r["model_input"]["original_user_request"].split()))
    prior_request_actions = {digest([r["model_input"]["original_user_request"], r["model_input"]["proposed_action"]]) for r in existing + prior}
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", COMMIT, "runs"],
                                    cwd=args.upstream, text=True).splitlines()
    planned = []
    for path in files:
        parts = path.split("/")
        if len(parts) != 6:
            continue
        _, model, suite, task, attack, filename = parts
        if model not in MODELS or suite != "workspace" or task not in train_tasks:
            continue
        if not filename.endswith(".json") or path in prior_source_paths:
            continue
        planned.append(path)
    # Deterministic source sampling before inspecting outcomes: at most four
    # traces per task/model/attack. No model or rule predictions used.
    source_groups = defaultdict(list)
    for path in planned:
        source_groups[tuple(path.split("/")[1:5])].append(path)
    planned = sorted(p for values in source_groups.values()
                     for p in sorted(values, key=lambda p: digest([4266, p]))[:4])
    print(f"Planned {len(planned)} pinned traces from {len(MODELS)} models", flush=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    def download(path):
        target = args.cache / path
        try:
            if not target.exists():
                if not args.fetch:
                    return path, "not_cached"
                url = f"https://raw.githubusercontent.com/ethz-spylab/agentdojo/{COMMIT}/{path}"
                request = Request(url, headers={"User-Agent": "DSA4266-public-data-review/1.0"})
                with urlopen(request, timeout=35) as response:
                    body = response.read()
                json.loads(body)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
            return path, None
        except Exception as error:
            return path, type(error).__name__

    fetched, failures = [], []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download, p) for p in planned]
        completed = 0
        for future in as_completed(futures):
            path, error = future.result()
            completed += 1
            if completed % 50 == 0: print(f"Read/downloaded {completed}/{len(planned)}", flush=True)
            if error:
                failures.append({"path": path, "error": error})
            else:
                fetched.append(path)
    print(f"Fetched/read {len(fetched)}/{len(planned)} public traces", flush=True)
    pool, skipped = {}, Counter()
    for path in sorted(fetched):
        body = (args.cache / path).read_bytes()
        trace = json.loads(body)
        if trace.get("error") is not None:
            skipped["episode_error"] += 1
            continue
        task = path.split("/")[3]
        group = "workspace:" + task
        source = {"upstream_path": path, "upstream_commit": COMMIT,
                  "source_url": f"{REPO_URL}/blob/{COMMIT}/{path}",
                  "source_sha256": hashlib.sha256(body).hexdigest(),
                  "agent_model": path.split("/")[1], "attack_type": path.split("/")[4],
                  "injection_task_id": trace.get("injection_task_id"),
                  "episode_security": trace.get("security"), "episode_utility": trace.get("utility")}
        try:
            extracted = list(extract(trace, source))
        except (ValueError, KeyError, TypeError) as error:
            skipped["unsupported_trace_format_" + type(error).__name__] += 1
            continue
        for m, parent in extracted:
            if " ".join(m["original_user_request"].split()) not in existing_requests[group]:
                skipped["request_mismatch_with_existing_task"] += 1
                continue
            fp = fingerprint(m)
            if fp in blocked:
                skipped["exact_duplicate_of_existing_502_or_prior_120"] += 1
                continue
            if fp not in pool:
                pool[fp] = {"candidate_id": "public-" + fp, "label": None, "model_input": m,
                            "model_input_fingerprint": fp, "source_type": "observed_public_benchmark",
                            "task_group_id": group, "task_family_id": assignments[group]["task_family_id"],
                            "existing_split": "train", "permitted_use": "train_development_only_after_review",
                            "same_request_action_in_prior_622": digest([m["original_user_request"], m["proposed_action"]]) in prior_request_actions,
                            "parent_sources": []}
            pool[fp]["parent_sources"].append(parent)
    strata = defaultdict(list)
    for row in pool.values():
        parents = row["parent_sources"]
        stratum = ("normal_episode" if any(p["attack_type"] == "none" for p in parents) else
                   "attacked_episode_security_false" if any(p["episode_security"] is False for p in parents) else
                   "attacked_episode_other")
        row["sampling_stratum"] = stratum
        strata[stratum].append(row)
    # Prefer new request/action combinations. First take at most one per
    # combination and one per source episode, then relax episode exclusivity.
    selected, selected_ids, selected_ra, selected_sources = [], set(), set(), set()
    for allow_prior_ra, allow_same_episode, allow_repeat_ra in (
        (False, False, False), (False, True, False),
        (True, True, False), (True, True, True)):
        groups = defaultdict(list)
        for row in pool.values():
            parent = row["parent_sources"][0]
            groups[(row["task_group_id"], row["model_input"]["proposed_action"]["tool_name"],
                    parent["agent_model"], parent["attack_type"])].append(row)
        for values in groups.values():
            values.sort(key=lambda r: digest([4266, r["candidate_id"]]))
        while groups and len(selected) < args.max_candidates:
            for key in sorted(list(groups), key=lambda k: digest([4266, k])):
                row = groups[key].pop(0)
                if not groups[key]: del groups[key]
                m = row["model_input"]
                ra = digest([m["original_user_request"], m["proposed_action"]])
                sources = {p["upstream_path"] for p in row["parent_sources"]}
                if row["candidate_id"] in selected_ids: continue
                if not allow_prior_ra and row["same_request_action_in_prior_622"]: continue
                if not allow_repeat_ra and ra in selected_ra: continue
                if not allow_same_episode and sources & selected_sources: continue
                row["selection_stage"] = {"prior_request_action_allowed":allow_prior_ra,
                    "same_episode_allowed":allow_same_episode,"repeat_request_action_allowed":allow_repeat_ra}
                selected.append(row); selected_ids.add(row["candidate_id"])
                selected_ra.add(ra); selected_sources.update(sources)
                if len(selected) == args.max_candidates: break
    if len(selected) != args.max_candidates:
        raise ValueError(f"Insufficient unique candidates: {len(selected)} / {args.max_candidates}; no partial delivery")
    selected.sort(key=lambda r: r["candidate_id"])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "annotation_candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8")
    review = [{"candidate_id": r["candidate_id"], "label": None, "review_status": "pending", "reviewer": "",
               "reason": "", "evidence": "", "task_group_id": r["task_group_id"],
               "sampling_stratum": r["sampling_stratum"],
               "note": "Episode security is triage metadata, never a call-level label. Review request and pre-call evidence."}
              for r in selected]
    (out / "review_metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in review), encoding="utf-8")
    with (out / "review_sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["row_no", "candidate_id", "label", "review_status", "reviewer", "reason", "evidence", "task_group_id",
                  "original_user_request", "tool_name", "arguments", "observation_preview", "full_context_file", "source_url"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        (out / "review_cases").mkdir()
        for row_no, r in enumerate(selected, 1):
            m = r["model_input"]
            full_path = f"review_cases/{row_no:03d}.json"
            review_case = {"candidate_id": r["candidate_id"], "label": None, "review_status": "pending",
                           "reviewer": "", "reason": "", "evidence": "", "model_input": m}
            (out / full_path).write_text(json.dumps(review_case, indent=2, ensure_ascii=False), encoding="utf-8")
            writer.writerow({"row_no": row_no, "candidate_id": r["candidate_id"], "label": "", "review_status": "pending",
                             "reviewer": "", "reason": "", "evidence": "", "task_group_id": r["task_group_id"],
                             "original_user_request": m["original_user_request"], "tool_name": m["proposed_action"]["tool_name"],
                             "arguments": json.dumps(m["proposed_action"]["arguments"], ensure_ascii=False),
                             "observation_preview": json.dumps(m["observations_before_call"], ensure_ascii=False)[:1500],
                             "full_context_file": full_path,
                             "source_url": r["parent_sources"][0]["source_url"]})
    used_paths = sorted({p["upstream_path"] for r in selected for p in r["parent_sources"]})
    source_manifest = []
    for path in used_paths:
        target = out / "sources" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.cache / path, target)
        source_manifest.append({"path": "sources/"+path, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                                "url": f"{REPO_URL}/blob/{COMMIT}/{path}"})
    (out / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    source_refs = Counter(p["upstream_path"] for r in selected for p in r["parent_sources"])
    stats = {"upstream_commit": COMMIT, "seed": 4266, "planned_traces": len(planned),
             "available_traces": len(fetched), "download_failures": failures, "skipped": dict(skipped),
             "unique_candidate_pool": len(pool), "delivered_candidates": len(selected),
             "label_counts": {"pending_null": len(selected)}, "tools": dict(Counter(r["model_input"]["proposed_action"]["tool_name"] for r in selected)),
             "sampling_strata": dict(Counter(r["sampling_stratum"] for r in selected)),
             "task_groups": sorted({r["task_group_id"] for r in selected}),
             "task_families": sorted({r["task_family_id"] for r in selected}),
             "source_traces_included": len(used_paths),
             "agent_models": sorted({p["agent_model"] for r in selected for p in r["parent_sources"]}),
             "attack_types": sorted({p["attack_type"] for r in selected for p in r["parent_sources"]}),
             "prior_622_request_action_overlap_rows": sum(r["same_request_action_in_prior_622"] for r in selected),
             "fixed_attack_email_in_action_rows": sum("mark.black-2134@gmail.com" in json.dumps(r["model_input"]["proposed_action"]) for r in selected),
             "within_batch_unique_request_actions": len(selected_ra),
             "within_batch_source_reference_count": sum(source_refs.values()),
             "within_batch_reused_source_count": sum(v > 1 for v in source_refs.values()),
             "rows_selected_after_relaxing_episode_uniqueness": sum(r["selection_stage"]["same_episode_allowed"] for r in selected),
             "prior_120_exact_overlap": 0,
             "prior_120_source_episode_overlap": 0,
             "prior_120_source_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
             "existing_502_source_sha256": hashlib.sha256(existing_path.read_bytes()).hexdigest(),
             "use_restriction": "train/development only after human review; no old validation/test tasks collected",
             "selection_note": "Deterministic source cap of 4 per task/model/attack; diversity round-robin; prefer novel request/action and distinct episodes. No rule or learned-model scores used. Episode strata are not call labels."}
    (out / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()

