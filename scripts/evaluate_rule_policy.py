#!/usr/bin/env python3
"""Evaluate the fixed lexical policy baseline on dataset-502-v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tool_call_auditor.modeling import binary_metrics, load_split, select_threshold
from tool_call_auditor.modeling.rule_policy import policy_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/splits/dataset_502_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/rule_policy"))
    parser.add_argument("--minimum-recall", type=float, default=0.80)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = {
        split: load_split(args.data_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    scores = {}
    features = {}
    for split, split_rows in rows.items():
        evaluated = [policy_score(row.model_input) for row in split_rows]
        scores[split] = [item[0] for item in evaluated]
        features[split] = [item[1] for item in evaluated]

    validation_labels = [row.label for row in rows["validation"]]
    threshold, selection = select_threshold(
        validation_labels, scores["validation"], minimum_recall=args.minimum_recall
    )
    metrics = {
        split: binary_metrics(
            [row.label for row in split_rows], scores[split], threshold
        )
        for split, split_rows in rows.items()
    }
    report = {
        "policy": {
            "injection_cue_present": 0.20,
            "tool_intent_mismatch": 0.35,
            "action_email_not_explicit_in_request": 0.10,
            "action_target_repeated_in_injection": 0.35,
        },
        "threshold_selected_on": "validation",
        "selected_threshold": threshold,
        "validation_selection": selection,
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    for split, split_rows in rows.items():
        with (args.output_dir / f"{split}_predictions.jsonl").open("w") as handle:
            for row, score, row_features in zip(
                split_rows, scores[split], features[split], strict=True
            ):
                handle.write(
                    json.dumps(
                        {
                            "candidate_id": row.candidate_id,
                            "label": row.label,
                            "probability_label_1": score,
                            "prediction": int(score >= threshold),
                            "features": row_features,
                        }
                    )
                    + "\n"
                )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
