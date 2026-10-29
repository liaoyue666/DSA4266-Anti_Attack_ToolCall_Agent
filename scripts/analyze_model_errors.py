#!/usr/bin/env python3
"""Join held-out predictions to review metadata for error analysis only."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/dataset_502_v1/labeled_calls.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}
    with args.annotations.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metadata[row["candidate_id"]] = row
    predictions = [json.loads(line) for line in args.predictions.open(encoding="utf-8")]

    joined = []
    for prediction in predictions:
        source = metadata[prediction["candidate_id"]]
        action = source["model_input"]["proposed_action"]
        joined.append(
            {
                **prediction,
                "source_type": source["source_type"],
                "task_group_id": source["task_group_id"],
                "injection_present": source["injection_present"],
                "tool_name": action["tool_name"],
                "error_type": (
                    "correct"
                    if prediction["prediction"] == prediction["label"]
                    else "false_block"
                    if prediction["label"] == 0
                    else "missed_unauthorized"
                ),
            }
        )

    fields = list(joined[0])
    with (args.output_dir / "errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(joined)

    breakdown = {}
    for field in ("source_type", "injection_present", "tool_name", "task_group_id"):
        groups = defaultdict(list)
        for row in joined:
            groups[str(row[field])].append(row)
        breakdown[field] = {}
        for value, group in sorted(groups.items()):
            labels = [row["label"] for row in group]
            predictions_for_group = [row["prediction"] for row in group]
            tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions_for_group))
            fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions_for_group))
            fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions_for_group))
            tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions_for_group))
            breakdown[field][value] = {
                "rows": len(group),
                "label_0": labels.count(0),
                "label_1": labels.count(1),
                "unauthorized_recall": safe_divide(tp, tp + fn),
                "false_block_rate": safe_divide(fp, fp + tn),
                "mean_score": sum(row["probability_label_1"] for row in group) / len(group),
            }
    (args.output_dir / "breakdown.json").write_text(json.dumps(breakdown, indent=2) + "\n")
    print(json.dumps(breakdown, indent=2))


if __name__ == "__main__":
    main()
