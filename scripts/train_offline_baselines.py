#!/usr/bin/env python3
"""Train majority and TF-IDF logistic-regression verifier baselines."""

from __future__ import annotations

import argparse
import json
import hashlib
import importlib.metadata
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from tool_call_auditor.modeling import (
    binary_metrics,
    load_split,
    select_threshold,
    serialize_model_input,
)
from tool_call_auditor.modeling.ablations import ABLATIONS, ablate_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/splits/dataset_502_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/offline_baselines"))
    parser.add_argument("--seed", type=int, default=4266)
    parser.add_argument("--minimum-recall", type=float, default=0.80)
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--development-only", action="store_true",
                        help="Read only train/validation; do not open or evaluate test.")
    return parser.parse_args()


def build_tfidf(seed: int) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=seed,
        solver="liblinear",
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def write_predictions(path: Path, rows, probabilities: np.ndarray, threshold: float) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row, probability in zip(rows, probabilities, strict=True):
            output = {
                "candidate_id": row.candidate_id,
                "label": row.label,
                "probability_label_1": float(probability),
                "prediction": int(probability >= threshold),
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory is not empty; select a new directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_names = ("train", "validation") if args.development_only else ("train", "validation", "test")
    splits = {
        name: ablate_rows(load_split(args.data_dir / f"{name}.jsonl"), args.ablation)
        for name in split_names
    }
    texts = {
        name: [serialize_model_input(row.model_input) for row in rows]
        for name, rows in splits.items()
    }
    labels = {name: np.asarray([row.label for row in rows]) for name, rows in splits.items()}

    majority_label = int(np.mean(labels["train"]) >= 0.5)
    majority_probability = float(majority_label)
    majority_results = {
        name: binary_metrics(y, np.full(len(y), majority_probability), 0.5)
        for name, y in labels.items()
    }

    model = build_tfidf(args.seed)
    model.fit(texts["train"], labels["train"])
    probabilities = {
        name: model.predict_proba(split_texts)[:, 1]
        for name, split_texts in texts.items()
    }
    threshold, validation_selection = select_threshold(
        labels["validation"],
        probabilities["validation"],
        minimum_recall=args.minimum_recall,
    )
    tfidf_results = {
        name: binary_metrics(labels[name], probabilities[name], threshold)
        for name in splits
    }

    report = {
        "ablation": args.ablation,
        "evaluated_splits": list(split_names),
        "input_sha256": {name: hashlib.sha256((args.data_dir / f"{name}.jsonl").read_bytes()).hexdigest()
                         for name in split_names},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "joblib")},
        "data_dir": str(args.data_dir),
        "seed": args.seed,
        "threshold_policy": {
            "selected_on": "validation",
            "minimum_unauthorized_recall": args.minimum_recall,
            "selected_threshold": threshold,
            "validation_selection": validation_selection,
        },
        "majority": {"predicted_label": majority_label, "metrics": majority_results},
        "tfidf_logistic_regression": {"metrics": tfidf_results},
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    joblib.dump(model, args.output_dir / "tfidf_logistic_regression.joblib")
    for name, rows in splits.items():
        write_predictions(args.output_dir / f"{name}_predictions.jsonl", rows, probabilities[name], threshold)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
