#!/usr/bin/env python3
"""Merge the reviewed public Workspace datasets and build grouped CV folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def summarize(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "labels": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        "sources": dict(sorted(Counter(row["source_dataset"] for row in rows).items())),
        "tools": dict(
            sorted(
                Counter(
                    row["model_input"]["proposed_action"]["tool_name"] for row in rows
                ).items()
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public-120",
        type=Path,
        default=Path(
            "data/annotations/agentdojo_workspace_120_v1/"
            "workspace120-lzs-2026-10-30-reviewed.jsonl"
        ),
    )
    parser.add_argument(
        "--public-120-provenance",
        type=Path,
        default=Path("data/annotations/agentdojo_workspace_120_v1/provenance.jsonl"),
    )
    parser.add_argument(
        "--public-400-reviewed",
        type=Path,
        default=Path("../四人400条复核合并/reviewed.jsonl"),
    )
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=Path("data/manifests/agentdojo_workspace_120_cv_folds.json"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/annotations/public_workspace_509_v1"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits/public_workspace_509_v1_cv"),
    )
    args = parser.parse_args()

    public_120 = read_jsonl(args.public_120)
    public_400 = read_jsonl(args.public_400_reviewed)
    provenance_120 = read_jsonl(args.public_120_provenance)
    family_by_group: dict[str, str] = {}
    for row in provenance_120:
        group = row["task_group_id"]
        family = row["task_family_id"]
        if group in family_by_group and family_by_group[group] != family:
            raise ValueError(f"Conflicting task family for {group}")
        family_by_group[group] = family

    if len(public_120) != 120 or len(public_400) != 389:
        raise ValueError("Expected 120 reviewed rows and 389 confirmed reviewed rows")
    merged = []
    for source, rows in (
        ("public_workspace_120_v1", public_120),
        ("public_workspace_400_v1_confirmed", public_400),
    ):
        for original in rows:
            if original.get("label") not in (0, 1):
                raise ValueError(f"Non-binary label in {source}: {original.get('candidate_id')}")
            row = dict(original)
            row["source_dataset"] = source
            merged.append(row)
    merged.sort(key=lambda row: row["candidate_id"])

    ids = [row["candidate_id"] for row in merged]
    fingerprints = [
        json.dumps(row["model_input"], ensure_ascii=False, sort_keys=True)
        for row in merged
    ]
    if len(merged) != 509 or len(set(ids)) != 509:
        raise ValueError("Merged dataset must contain 509 unique candidate IDs")
    if len(set(fingerprints)) != 509:
        raise ValueError("Merged dataset contains duplicate model inputs")
    missing_groups = sorted({row["task_group_id"] for row in merged} - family_by_group.keys())
    if missing_groups:
        raise ValueError(f"Missing task-family mapping for {missing_groups}")

    fold_config = json.loads(args.fold_manifest.read_text(encoding="utf-8"))
    family_to_fold = {
        family: fold
        for fold, families in fold_config["folds"].items()
        for family in families
    }
    assignments = []
    for row in merged:
        family = family_by_group[row["task_group_id"]]
        assignments.append(
            {
                "candidate_id": row["candidate_id"],
                "source_dataset": row["source_dataset"],
                "task_group_id": row["task_group_id"],
                "task_family_id": family,
                "validation_fold": family_to_fold[family],
                "label": row["label"],
                "tool_name": row["model_input"]["proposed_action"]["tool_name"],
            }
        )

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dataset_dir / "dataset.jsonl"
    write_jsonl(dataset_path, merged)
    stats = {
        "dataset_id": "public-workspace-509-v1",
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "protocol": "four-fold leave-task-families-out cross-validation",
        "excluded": {"public_workspace_400_v1_unresolved": 11},
        "overall": summarize(merged),
        "folds": {},
    }
    assignment_by_id = {row["candidate_id"]: row for row in assignments}
    for fold in fold_config["folds"]:
        validation = [
            row for row in merged if assignment_by_id[row["candidate_id"]]["validation_fold"] == fold
        ]
        train = [
            row for row in merged if assignment_by_id[row["candidate_id"]]["validation_fold"] != fold
        ]
        train_families = {
            assignment_by_id[row["candidate_id"]]["task_family_id"] for row in train
        }
        validation_families = {
            assignment_by_id[row["candidate_id"]]["task_family_id"] for row in validation
        }
        if train_families & validation_families:
            raise AssertionError("Task-family leakage detected")
        if {row["label"] for row in validation} != {0, 1}:
            raise AssertionError(f"{fold} does not contain both labels")
        fold_dir = args.split_dir / fold
        fold_dir.mkdir(exist_ok=True)
        write_jsonl(fold_dir / "train.jsonl", train)
        write_jsonl(fold_dir / "validation.jsonl", validation)
        stats["folds"][fold] = {
            "validation_families": sorted(validation_families),
            "train": summarize(train),
            "validation": summarize(validation),
        }

    with (args.split_dir / "fold_assignments.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignments[0]))
        writer.writeheader()
        writer.writerows(assignments)
    (args.split_dir / "cross_validation_statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.dataset_dir / "README.md").write_text(
        "# Public Workspace 509 v1\n\n"
        "This development-only dataset merges 120 reviewed public AgentDojo calls with "
        "389 confirmed rows from the team-reviewed public Workspace 400 set. The 11 "
        "unresolved rows are excluded. `source_dataset` preserves row provenance. "
        "Evaluation uses task-family-isolated cross-validation; this dataset is not an "
        "independent benchmark test set.\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
