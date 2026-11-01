#!/usr/bin/env python3
"""Run fixed-threshold baselines over the four public-call CV folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tool_call_auditor.modeling import binary_metrics, load_split, serialize_model_input
from tool_call_auditor.modeling.rule_policy import policy_score
from train_offline_baselines import build_tfidf


def aggregate(fold_results: list[dict], all_labels: list[int], all_scores: list[float]) -> dict:
    metrics = binary_metrics(all_labels, all_scores, 0.5)
    metric_names = (
        "unauthorized_precision",
        "unauthorized_recall",
        "unauthorized_f1",
        "false_block_rate",
        "pr_auc",
    )
    return {
        "out_of_fold_micro": metrics,
        "fold_macro_mean": {
            name: float(np.mean([fold["metrics"][name] for fold in fold_results]))
            for name in metric_names
        },
        "fold_macro_std": {
            name: float(np.std([fold["metrics"][name] for fold in fold_results]))
            for name in metric_names
        },
        "folds": fold_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/splits/agentdojo_workspace_120_cv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/agentdojo_workspace_120_cv")
    )
    parser.add_argument("--seed", type=int, default=4266)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_outputs = {name: {"folds": [], "labels": [], "scores": []} for name in (
        "majority", "tfidf_logistic_regression", "rule_policy"
    )}
    prediction_rows = []
    for fold_dir in sorted(path for path in args.data_dir.glob("fold_*") if path.is_dir()):
        train = load_split(fold_dir / "train.jsonl")
        validation = load_split(fold_dir / "validation.jsonl")
        train_labels = np.asarray([row.label for row in train])
        validation_labels = [row.label for row in validation]
        train_text = [serialize_model_input(row.model_input) for row in train]
        validation_text = [serialize_model_input(row.model_input) for row in validation]
        train_tools = {
            row.model_input["proposed_action"]["tool_name"] for row in train
        }

        majority_label = int(np.mean(train_labels) >= 0.5)
        majority_scores = np.full(len(validation), float(majority_label))
        tfidf = build_tfidf(args.seed)
        tfidf.fit(train_text, train_labels)
        tfidf_scores = tfidf.predict_proba(validation_text)[:, 1]
        rule_scores = np.asarray([policy_score(row.model_input)[0] for row in validation])
        fold_scores = {
            "majority": majority_scores,
            "tfidf_logistic_regression": tfidf_scores,
            "rule_policy": rule_scores,
        }
        for model_name, scores in fold_scores.items():
            metrics = binary_metrics(validation_labels, scores, 0.5)
            model_outputs[model_name]["folds"].append(
                {"fold": fold_dir.name, "metrics": metrics}
            )
            model_outputs[model_name]["labels"].extend(validation_labels)
            model_outputs[model_name]["scores"].extend(scores.tolist())
        for index, row in enumerate(validation):
            tool_name = row.model_input["proposed_action"]["tool_name"]
            prediction_rows.append(
                {
                    "candidate_id": row.candidate_id,
                    "fold": fold_dir.name,
                    "label": row.label,
                    "tool_name": tool_name,
                    "tool_seen_in_fold_train": tool_name in train_tools,
                    **{
                        f"{name}_score": float(scores[index])
                        for name, scores in fold_scores.items()
                    },
                }
            )

    expected_ids = {
        row["candidate_id"]
        for row in __import__("csv").DictReader(
            (args.data_dir / "fold_assignments.csv").open(encoding="utf-8")
        )
    }
    predicted_ids = {row["candidate_id"] for row in prediction_rows}
    if predicted_ids != expected_ids or len(prediction_rows) != len(expected_ids):
        raise AssertionError("Each assigned row must appear in validation exactly once")
    report = {
        "protocol": "four-fold leave-task-families-out cross-validation",
        "threshold": 0.5,
        "threshold_selected_on_validation": False,
        "models": {
            name: aggregate(values["folds"], values["labels"], values["scores"])
            for name, values in model_outputs.items()
        },
    }
    seen_rows = [row for row in prediction_rows if row["tool_seen_in_fold_train"]]
    report["seen_tool_subset"] = {
        "rows": len(seen_rows),
        "labels": {
            "0": sum(row["label"] == 0 for row in seen_rows),
            "1": sum(row["label"] == 1 for row in seen_rows),
        },
        "models": {
            name: binary_metrics(
                [row["label"] for row in seen_rows],
                [row[f"{name}_score"] for row in seen_rows],
                0.5,
            )
            for name in model_outputs
        },
    }
    (args.output_dir / "baseline_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "out_of_fold_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
