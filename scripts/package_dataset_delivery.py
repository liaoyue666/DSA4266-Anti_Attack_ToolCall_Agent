from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdojo.task_suite import get_suite

from scripts.build_constructed_candidates import (
    _load_phase_records,
    _model_input_fingerprint,
    _observed_model_input,
    _read_catalog,
    _read_jsonl,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONSTRUCTION_DIR = Path("data/constructed/delivery-001")
DEFAULT_REPORT = DEFAULT_CONSTRUCTION_DIR / "construction_report.json"
DEFAULT_OUTPUT_DIR = Path("data/delivery/dataset-502-v1")
CATALOG_PATH = Path("data/manifests/agentdojo_workspace_v1.2.2_catalog.json")
CONSTRUCTION_CODE_PATH = Path("scripts/build_constructed_candidates.py")


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _source_reference(call: dict, episode: dict, trace_path: Path) -> dict:
    return {
        "call_id": call["call_id"],
        "episode_id": call["episode_id"],
        "source_id": call["source_id"],
        "source_name": call["_source_name"],
        "source_export_path": call["_source_export_path"],
        "source_trace_path": trace_path.as_posix(),
        "task_group_id": call["task_group_id"],
        "user_task_id": call["user_task_id"],
        "injection_task_id": call.get("injection_task_id"),
    }


def _observed_rows(linked: list[tuple[dict, dict, Path]]) -> tuple[list[dict], dict]:
    grouped: dict[str, list[tuple[dict, dict, Path]]] = defaultdict(list)
    for call, episode, trace_path in linked:
        model_input = _observed_model_input(call)
        fingerprint = _model_input_fingerprint(model_input)
        grouped[fingerprint].append((call, episode, trace_path))

    rows = []
    for fingerprint, source_rows in sorted(grouped.items()):
        call, episode, trace_path = source_rows[0]
        references = [
            _source_reference(ref_call, ref_episode, ref_trace)
            for ref_call, ref_episode, ref_trace in source_rows
        ]
        task_group_ids = sorted(
            {str(reference["task_group_id"]) for reference in references}
        )
        model_input = _observed_model_input(call)
        rows.append(
            {
                "candidate_id": f"observed-{fingerprint}",
                "model_input_fingerprint": fingerprint,
                "source_type": "observed",
                "task_group_id": task_group_ids[0],
                "related_task_group_ids": task_group_ids,
                "parent_sources": references,
                "near_duplicate_group_id": None,
                "model_input": model_input,
                "label": None,
            }
        )
    return rows, {
        "raw_call_count": len(linked),
        "unique_call_count": len(rows),
        "duplicate_rows_removed": len(linked) - len(rows),
        "duplicate_fingerprint_group_count": sum(
            len(source_rows) > 1 for source_rows in grouped.values()
        ),
    }


def _annotation_row(row: dict) -> dict:
    model_input = row["model_input"]
    if set(model_input) != {
        "original_user_request",
        "observations_before_call",
        "proposed_action",
    }:
        raise ValueError(f"Unexpected model input keys for {row['candidate_id']}")
    if set(model_input["proposed_action"]) != {"tool_name", "arguments"}:
        raise ValueError(f"Unexpected proposed action keys for {row['candidate_id']}")

    parent = row.get("parent") or {}
    task_group_id = row.get("task_group_id") or parent.get("task_group_id")
    parent_sources = (
        row.get("parent_sources")
        or row.get("parent_references")
        or ([parent] if parent else [])
    )
    if not task_group_id:
        raise ValueError(f"Missing task group for {row['candidate_id']}")
    task_group_ids = sorted(
        {
            str(reference.get("task_group_id") or task_group_id)
            for reference in parent_sources
        }
    )
    return {
        "candidate_id": row["candidate_id"],
        "model_input_fingerprint": _model_input_fingerprint(model_input),
        "source_type": row["source_type"],
        "task_group_id": task_group_ids[0],
        "related_task_group_ids": task_group_ids,
        "parent_sources": parent_sources,
        "near_duplicate_group_id": row.get("near_duplicate_group_id"),
        "model_input": model_input,
        "label": None,
    }


def _review_rows(
    constructed_rows: list[dict], source_review_rows: list[dict]
) -> list[dict]:
    constructed_by_id = {row["candidate_id"]: row for row in constructed_rows}
    if len(constructed_by_id) != len(constructed_rows):
        raise ValueError("Constructed candidate IDs are not unique")

    review_by_id: dict[str, dict] = {}
    for review in source_review_rows:
        candidate_id = review["candidate_id"]
        if candidate_id in review_by_id:
            raise ValueError(f"Duplicate review record for {candidate_id}")
        review_by_id[candidate_id] = review

    matched = []
    for candidate_id in sorted(constructed_by_id):
        review = review_by_id.get(candidate_id)
        if review is None:
            raise ValueError(f"Missing review record for delivered candidate {candidate_id}")
        if review.get("label") is not None:
            raise ValueError(f"Review label is not null for {candidate_id}")
        matched.append(review)
    return matched


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_version() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "working_tree_dirty": bool(status),
        "dirty_paths": [line[3:] for line in status.splitlines()],
    }


def _manifest_sources(
    export_roots: list[Path],
    construction_dir: Path,
    report_path: Path,
) -> list[dict]:
    paths: set[Path] = set()
    for root in export_roots:
        for path in root.rglob("*"):
            if path.is_file() and not path.name.startswith("."):
                paths.add(path)
    paths.update(
        {
            construction_dir / "constructed_candidates.jsonl",
            construction_dir / "expected_unauthorized_review.jsonl",
            report_path,
            CATALOG_PATH,
            CONSTRUCTION_CODE_PATH,
            Path(__file__),
        }
    )
    missing = sorted(path for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")
    return [
        {
            "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(item.resolve() for item in paths)
    ]


def _statistics(
    observed_rows: list[dict],
    constructed_rows: list[dict],
    annotation_rows: list[dict],
    reviews: list[dict],
    duplicate_summary: dict,
    report: dict,
) -> dict:
    observed_fingerprints = {
        row["model_input_fingerprint"] for row in observed_rows
    }
    constructed_fingerprints = {
        row["model_input_fingerprint"] for row in constructed_rows
    }
    if len(constructed_fingerprints) != len(constructed_rows):
        raise ValueError("Constructed model-input fingerprints are not unique")
    cross_source_duplicates = observed_fingerprints & constructed_fingerprints
    if cross_source_duplicates:
        raise ValueError(
            f"Constructed inputs duplicate observed inputs: {len(cross_source_duplicates)}"
        )

    tool_counts = {
        source: dict(
            sorted(
                Counter(
                    row["model_input"]["proposed_action"]["tool_name"]
                    for row in rows
                ).items()
            )
        )
        for source, rows in (
            ("observed", observed_rows),
            ("constructed", constructed_rows),
        )
    }
    task_counts = {
        source: dict(
            sorted(
                Counter(
                    group
                    for row in rows
                    for group in row["related_task_group_ids"]
                ).items()
            )
        )
        for source, rows in (
            ("observed", observed_rows),
            ("constructed", constructed_rows),
        )
    }
    near_groups = Counter(
        row["near_duplicate_group_id"]
        for row in constructed_rows
        if row.get("near_duplicate_group_id")
    )
    method_counts = Counter(review["construction_method"] for review in reviews)
    return {
        "observed": {
            **duplicate_summary,
            "task_group_coverage": sorted(
                {
                    group
                    for row in observed_rows
                    for group in row["related_task_group_ids"]
                }
            ),
            "tool_counts": tool_counts["observed"],
        },
        "constructed": {
            "candidate_count": len(constructed_rows),
            "expected_unauthorized_intent_count": len(reviews),
            "task_group_coverage": sorted(
                {
                    group
                    for row in constructed_rows
                    for group in row["related_task_group_ids"]
                }
            ),
            "task_group_candidate_counts": task_counts["constructed"],
            "tool_counts": tool_counts["constructed"],
            "construction_method_counts": dict(sorted(method_counts.items())),
            "near_duplicate_group_count": len(near_groups),
            "near_duplicate_group_sizes": dict(sorted(near_groups.items())),
            "largest_near_duplicate_group_size": max(near_groups.values(), default=0),
            "not_confirmed_label_1": True,
            "excluded_unmatched_review_record_count": report[
                "expected_unauthorized_review_count"
            ]
            - len(reviews),
        },
        "total": {
            "annotation_candidate_count": len(annotation_rows),
            "unique_candidate_id_count": len(
                {row["candidate_id"] for row in annotation_rows}
            ),
            "unique_model_input_fingerprint_count": len(
                {row["model_input_fingerprint"] for row in annotation_rows}
            ),
            "cross_source_duplicate_count": len(cross_source_duplicates),
            "candidate_id_collision_count": 0,
            "label_counts": {"null": sum(row["label"] is None for row in annotation_rows)},
            "all_labels_null": all(row["label"] is None for row in annotation_rows),
            "candidate_rows_are_not_independent_task_count": True,
            "task_group_coverage": sorted(
                {
                    group
                    for row in annotation_rows
                    for group in row["related_task_group_ids"]
                }
            ),
            "task_group_counts_by_source": task_counts,
            "tool_counts_by_source": tool_counts,
        },
    }


def build_delivery(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite delivery directory: {output_dir}")

    report_path = DEFAULT_REPORT
    construction_dir = DEFAULT_CONSTRUCTION_DIR
    report = json.loads(report_path.read_text(encoding="utf-8"))
    export_roots = [Path(path) for path in report["input"]["export_roots"]]
    suite = get_suite("v1.2.2", "workspace")
    tool_schemas = {tool.name: tool.parameters for tool in suite.tools}
    catalog = _read_catalog(CATALOG_PATH)
    linked, input_summary = _load_phase_records(
        export_roots, catalog, tool_schemas
    )
    observed_rows, duplicate_summary = _observed_rows(linked)
    source_constructed = _read_jsonl(
        construction_dir / "constructed_candidates.jsonl"
    )
    source_reviews = _read_jsonl(
        construction_dir / "expected_unauthorized_review.jsonl"
    )

    if len(observed_rows) != report["observed_unique_model_input_count"]:
        raise ValueError("Observed unique count does not match construction report")
    if len(linked) != report["observed_raw_call_count"]:
        raise ValueError("Observed raw call count does not match construction report")
    if len(source_constructed) != report["new_constructed_count"]:
        raise ValueError("Constructed candidate count does not match construction report")
    if len(source_reviews) != report["expected_unauthorized_review_count"]:
        raise ValueError("Review row count does not match construction report")

    constructed_rows = source_constructed
    for row in constructed_rows:
        if row.get("source_type") != "constructed" or row.get("label") is not None:
            raise ValueError(f"Invalid constructed candidate record: {row.get('candidate_id')}")
        model_input = row["model_input"]
        if _model_input_fingerprint(model_input) != row["model_input_fingerprint"]:
            raise ValueError(f"Constructed fingerprint mismatch: {row['candidate_id']}")
        if not row.get("parent_references") or not row.get("parent", {}).get("call_id"):
            raise ValueError(f"Constructed candidate has no parent link: {row['candidate_id']}")
    reviews = _review_rows(constructed_rows, source_reviews)

    # Check parent references against the real calls before packaging.
    observed_call_ids = {call["call_id"] for call, _, _ in linked}
    for row in constructed_rows:
        if row["parent"]["call_id"] not in observed_call_ids:
            raise ValueError(f"Unknown parent call for {row['candidate_id']}")
        for reference in row["parent_references"]:
            if reference["call_id"] not in observed_call_ids:
                raise ValueError(
                    f"Unknown parent reference for {row['candidate_id']}: "
                    f"{reference['call_id']}"
                )

    observed_annotation_rows = [_annotation_row(row) for row in observed_rows]
    constructed_annotation_rows = [
        _annotation_row(row) for row in constructed_rows
    ]
    annotation_rows = [*observed_annotation_rows, *constructed_annotation_rows]
    if len({row["candidate_id"] for row in annotation_rows}) != len(annotation_rows):
        raise ValueError("Candidate IDs collide across delivery sources")
    if len({row["model_input_fingerprint"] for row in annotation_rows}) != len(
        annotation_rows
    ):
        raise ValueError("Model-input fingerprints collide across delivery sources")
    if any(row["label"] is not None for row in annotation_rows):
        raise ValueError("Non-null label found in annotation candidates")

    expected = report["total_exact_unique_candidate_rows"]
    if len(annotation_rows) != expected:
        raise ValueError(
            f"Delivery contains {len(annotation_rows)} rows; report expects {expected}"
        )

    stats = _statistics(
        observed_annotation_rows,
        constructed_annotation_rows,
        annotation_rows,
        reviews,
        duplicate_summary,
        report,
    )
    manifest = {
        "delivery_id": output_dir.name,
        "source_paths": {
            "observed_export_roots": [path.as_posix() for path in export_roots],
            "constructed_candidates": (
                construction_dir / "constructed_candidates.jsonl"
            ).as_posix(),
            "constructed_review_source": (
                construction_dir / "expected_unauthorized_review.jsonl"
            ).as_posix(),
            "construction_report": report_path.as_posix(),
            "catalog": CATALOG_PATH.as_posix(),
        },
        "source_file_hashes": _manifest_sources(
            export_roots, construction_dir, report_path
        ),
        "code_version": _code_version(),
        "observed_input_summary": input_summary,
        "expected_unauthorized_review_source_count": len(source_reviews),
        "included_review_count": len(reviews),
        "excluded_unmatched_review_count": len(source_reviews) - len(reviews),
    }
    readme = """# Dataset delivery: dataset-502-v1

This directory contains 502 unique annotation candidates: 382 observed calls
and 120 constructed candidates. These are candidates for human review, not
already annotated or adjudicated data. Every `label` is `null`.

The 120 constructed rows carry an `expected_unauthorized` review intent. That
intent is a hypothesis for review, **not a confirmed Label 1**. Constructed
rows were not proposed or executed by an agent and must not be included in
real-episode ASR calculations. Their construction explanations are isolated in
`review_metadata.jsonl` and are not part of `model_input`.

`observed_calls.jsonl` is deduplicated by the canonical model-input fingerprint
(original request, ordered pre-call observations, tool name, and arguments).
Each row retains all source call/episode references for exact duplicate source
rows. `constructed_calls.jsonl` is copied from the existing delivery-001 file;
it is not regenerated. Only review records whose IDs match those 120 delivered
constructed candidate IDs are included. The other 76 records in the source
196-row review list are excluded.

For every candidate, model input consists only of `original_user_request`,
`observations_before_call`, and `proposed_action` (`tool_name`, `arguments`).
Source IDs, task grouping, labels, duplicate groups, and all review metadata
remain outside that object.

Near-duplicate groups are retained and reported; 502 rows must not be
described as 502 independent tasks. When making any train/validation/test
split, keep every candidate with the same `task_group_id` together. For a
deduplicated row listing multiple `related_task_group_ids`, treat those groups
as linked and assign them to the same split as well; all constructed children
must remain with their parent task group. The final evaluation split still
requires team confirmation.

No API calls, collection runs, tool executions, or automatic labeling were
performed to prepare this package.
"""
    files = {
        "observed_calls.jsonl": _jsonl_bytes(observed_rows),
        "constructed_calls.jsonl": _jsonl_bytes(constructed_rows),
        "annotation_candidates.jsonl": _jsonl_bytes(annotation_rows),
        "review_metadata.jsonl": _jsonl_bytes(reviews),
        "dataset_statistics.json": (
            json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "source_manifest.json": (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "README.md": readme.encode("utf-8"),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    created = []
    try:
        for filename, content in files.items():
            path = output_dir / filename
            with path.open("xb") as stream:
                stream.write(content)
            created.append(path)
    except Exception:
        for path in created:
            path.unlink()
        output_dir.rmdir()
        raise
    return stats


def main() -> None:
    stats = build_delivery()
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
