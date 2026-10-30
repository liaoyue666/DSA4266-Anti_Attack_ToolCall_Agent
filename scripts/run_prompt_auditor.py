"""Dry-run by default; opt in to a bounded train/validation API audit."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics

from tool_call_auditor.modeling import binary_metrics, load_split
from tool_call_auditor.modeling.ablations import ABLATIONS, ablate_input
from tool_call_auditor.modeling.prompt_auditor import (
    AuditorConfig, PROMPT_VERSION, PromptAuditor, build_payload, request_key,
)

ROOT = Path(__file__).resolve().parents[1]


def summarize(predictions: list[dict], calls_used: int) -> dict:
    good = [r for r in predictions if r["status"] == "ok"]
    all_labeled = [r for r in predictions if r.get("label") in (0, 1)]
    valid_labeled = [r for r in good if r.get("label") in (0, 1)]
    current_usage = [r["usage_this_run"] for r in predictions if isinstance(r.get("usage_this_run"), dict)]
    latency = [r["provider_latency_seconds_this_run"] for r in predictions
               if r.get("provider_latency_seconds_this_run") is not None]
    token_totals = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [u[name] for u in current_usage if type(u.get(name)) is int]
        token_totals[name] = {"known_total": sum(values), "known_usage_records": len(values),
                              "unknown_api_calls": calls_used - len(values)}
    metrics = None
    if valid_labeled:
        metrics = binary_metrics([r["label"] for r in valid_labeled],
                                 [r["prediction"] for r in valid_labeled], .5)
        metrics.pop("pr_auc")  # Binary decisions are not ranked probabilities.
        metrics["score_type"] = "binary_decision_not_probability"
    return {
        "rows": len(predictions), "status_counts": dict(Counter(r["status"] for r in predictions)),
        "valid_response_coverage": len(good) / len(predictions) if predictions else 0,
        "api_calls_this_run": calls_used, "cache_hits": sum(r["cache_hit"] for r in predictions),
        "token_usage_this_run": token_totals, "monetary_cost": None,
        "mean_provider_latency_seconds_this_run": statistics.mean(latency) if latency else None,
        "conditional_metrics_on_valid_labeled_rows": metrics,
        "correct_valid_predictions_over_all_labeled_rows": (
            sum(r["prediction"] == r["label"] for r in valid_labeled) / len(all_labeled)
            if all_labeled else None),
        "error_policy": "Failures retain null prediction, counted in coverage; never silently converted to ALLOW/BLOCK.",
        "ranking_metrics": "Not reported: auditor emits categorical decisions, not calibrated/ranked scores.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/splits/dataset_502_v1")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results/prompt_auditor_cache_v1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-calls", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 1 or args.max_calls < 0:
        parser.error("limit must be positive and max-calls nonnegative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; select a new directory")
    source = args.data_dir / f"{args.split}.jsonl"
    rows = load_split(source)[:args.limit]  # Stable source order; no label-based sampling.
    config = AuditorConfig(model=args.model)
    prepared = [(row, ablate_input(row.model_input, args.ablation)) for row in rows]
    manifest = {
        "mode": "execute" if args.execute else "dry_run", "prompt_version": PROMPT_VERSION,
        "split": args.split, "ablation": args.ablation, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selected_rows": len(rows), "selection": "first N rows in existing split order; development smoke only",
        "selected_ids": [r.candidate_id for r in rows], "config": asdict(config),
        "max_api_calls": args.max_calls if args.execute else 0,
        "test_read_or_evaluated": False,
        "input_protocol": "Same observation selection/8000-character clipping as TF-IDF; structured JSON fields. More tokens than frozen encoder possible.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.execute:
        preview = [{"candidate_id": row.candidate_id, "request_key": request_key(build_payload(m, config), config),
                    "payload": build_payload(m, config)} for row, m in prepared]
        (args.output_dir / "request_preview.jsonl").write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preview), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return
    auditor = PromptAuditor(config, args.cache_dir, api_key=os.environ.get("DEEPSEEK_API_KEY", ""), max_calls=args.max_calls)
    predictions = []
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, model_input in prepared:
            result = {"candidate_id": row.candidate_id, "label": row.label,
                      "ablation": args.ablation, **auditor.audit(model_input)}
            predictions.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
    report = {**manifest, **summarize(predictions, auditor.calls_used)}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
