"""Reproduce sparse ablations and inspect a frozen policy without reading test."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
from tool_call_auditor.modeling import binary_metrics, load_split, select_threshold, serialize_model_input
from tool_call_auditor.modeling.rule_policy import policy_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.probe_dir = args.probe_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory is not empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data/splits/dataset_502_v1"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    reports = {}
    for mode in ("full", "no_request", "no_observations", "action_only"):
        out = args.output_dir / mode
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/train_offline_baselines.py"),
             "--data-dir", str(data_dir), "--output-dir", str(out),
             "--ablation", mode, "--development-only"],
            env=environment, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        (args.output_dir / f"{mode}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        result.check_returncode()
        reports[mode] = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        print("Finished TF-IDF:", mode, flush=True)
    validation = load_split(data_dir / "validation.jsonl")
    scores = [policy_score(row.model_input)[0] for row in validation]
    threshold, selection = select_threshold([row.label for row in validation], scores)
    probes = [json.loads(line) for line in (args.probe_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    review = {r["candidate_id"]: r for r in map(json.loads, (args.probe_dir / "review_metadata.jsonl").read_text(encoding="utf-8").splitlines())}
    sparse = joblib.load(args.output_dir / "full/tfidf_logistic_regression.joblib")
    sparse_scores = sparse.predict_proba([serialize_model_input(r["model_input"]) for r in probes])[:, 1]
    sparse_threshold = reports["full"]["threshold_policy"]["selected_threshold"]
    joined = []
    for row, sparse_score in zip(probes, sparse_scores, strict=True):
        expected = review[row["candidate_id"]]["expected_label"]
        score, features = policy_score(row["model_input"])
        joined.append({"candidate_id": row["candidate_id"], "expected_label_pending_review": expected,
                       "rule_score": score, "rule_prediction": int(score >= threshold), "rule_features": features,
                       "tfidf_score": float(sparse_score), "tfidf_prediction": int(sparse_score >= sparse_threshold)})
    y = [r["expected_label_pending_review"] for r in joined]
    report = {
        "scope": "Train/validation diagnostics only; validation is used for threshold selection, not an independent final estimate.",
        "test_read_or_evaluated": False,
        "versions": {x: importlib.metadata.version(x) for x in ("numpy", "scikit-learn", "joblib")},
        "data_sha256": {name: hashlib.sha256((data_dir / f"{name}.jsonl").read_bytes()).hexdigest()
                        for name in ("train", "validation")},
        "validation_tfidf": {m: r["tfidf_logistic_regression"]["metrics"]["validation"] for m, r in reports.items()},
        "validation_thresholds": {m: r["threshold_policy"]["selected_threshold"] for m, r in reports.items()},
        "rule_threshold": threshold, "rule_validation": selection,
        "probe_warning": "Synthetic deliberately targeted development probes with pending expected labels; not generalization/ASR results.",
        "probe_rule_vs_expected": binary_metrics(y, [r["rule_score"] for r in joined], threshold),
        "probe_tfidf_vs_expected": binary_metrics(y, sparse_scores, sparse_threshold),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "probe_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in joined), encoding="utf-8")
    # Use the existing error-analysis script, restricted to validation predictions.
    result = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_model_errors.py"),
                             "--predictions", str(args.output_dir / "full/validation_predictions.jsonl"),
                             "--annotations", str(ROOT / "data/annotations/dataset_502_v1/labeled_calls.jsonl"),
                             "--output-dir", str(args.output_dir / "validation_errors")],
                            env=environment, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    (args.output_dir / "validation_errors.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
