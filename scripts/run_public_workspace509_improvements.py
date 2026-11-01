#!/usr/bin/env python3
"""Run controlled frozen-encoder improvements on public-workspace-509-v1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from tool_call_auditor.modeling import binary_metrics, load_split, select_threshold
from tool_call_auditor.modeling.rule_policy import policy_features
from tool_call_auditor.modeling.serialization import _observation_text, select_observations
from tool_call_auditor.modeling.transformer_input import serialize_action
from train_frozen_transformer import DenseVerifier, choose_device, extract_embeddings


RULE_NAMES = (
    "injection_cue_present",
    "tool_intent_mismatch",
    "action_email_not_explicit_in_request",
    "action_target_repeated_in_injection",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def mean_pool(last_hidden_state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * expanded).sum(1) / expanded.sum(1).clamp(min=1)


def encode_texts(tokenizer, texts: list[str], max_length: int) -> TensorDataset:
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"])


def extract_text_embeddings(
    tokenizer,
    encoder,
    device: torch.device,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    output = []
    encoder.eval()
    with torch.inference_mode():
        for input_ids, mask in DataLoader(
            encode_texts(tokenizer, texts, max_length), batch_size=batch_size
        ):
            mask = mask.to(device)
            hidden = encoder(input_ids=input_ids.to(device), attention_mask=mask).last_hidden_state
            output.append(mean_pool(hidden, mask).cpu())
    return torch.cat(output)


def field_texts(model_input: dict) -> tuple[str, str, str]:
    request = str(model_input.get("original_user_request", ""))
    observations = select_observations(
        model_input.get("observations_before_call", []), max_observations=3
    )
    observation = "\n---\n".join(_observation_text(item) for item in observations)
    action = serialize_action(model_input.get("proposed_action", {}))
    return request, observation, action


def train_head(
    features: torch.Tensor,
    labels: np.ndarray,
    indices: list[int],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_mode: str,
) -> DenseVerifier:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = DenseVerifier(features.shape[1], dropout=0.3)
    selected_labels = labels[indices]
    positives = int(selected_labels.sum())
    negatives = len(selected_labels) - positives
    ratio = negatives / positives
    if weight_mode == "full":
        positive_weight = ratio
    elif weight_mode == "sqrt":
        positive_weight = math.sqrt(ratio)
    elif weight_mode == "none":
        positive_weight = 1.0
    else:
        raise ValueError(weight_mode)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    dataset = TensorDataset(
        features[indices], torch.tensor(selected_labels, dtype=torch.float32)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    for _ in range(epochs):
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
    return model


def predict(model: nn.Module, features: torch.Tensor, indices: list[int]) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(features[indices])).numpy()


def fit_ensemble(
    features: torch.Tensor,
    labels: np.ndarray,
    train_indices: list[int],
    score_indices: list[int],
    *,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_mode: str,
) -> np.ndarray:
    scores = []
    for seed in seeds:
        model = train_head(
            features,
            labels,
            train_indices,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_mode=weight_mode,
        )
        scores.append(predict(model, features, score_indices))
    return np.mean(np.asarray(scores), axis=0)


def inner_threshold(
    features: torch.Tensor,
    labels: np.ndarray,
    outer_train: list[int],
    families: list[str],
    *,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_mode: str,
    minimum_recall: float,
) -> tuple[float, dict]:
    by_family = defaultdict(list)
    for index in outer_train:
        by_family[families[index]].append(index)
    inner_scores = {}
    for held_out in sorted(by_family):
        validation = by_family[held_out]
        train = [index for index in outer_train if families[index] != held_out]
        if len(set(labels[train])) < 2:
            continue
        scores = fit_ensemble(
            features,
            labels,
            train,
            validation,
            seeds=seeds,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_mode=weight_mode,
        )
        inner_scores.update(zip(validation, scores.tolist(), strict=True))
    ordered = sorted(inner_scores)
    if len(ordered) != len(outer_train):
        raise AssertionError("Inner OOF thresholding did not cover the outer training set")
    return select_threshold(
        labels[ordered], [inner_scores[index] for index in ordered], minimum_recall=minimum_recall
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/annotations/public_workspace_509_v1/dataset.jsonl")
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("data/splits/public_workspace_509_v1_cv/fold_assignments.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/public_workspace_509_v1_cv/frozen_improvements"),
    )
    parser.add_argument("--model-name", default="distilbert/distilroberta-base")
    parser.add_argument("--seeds", type=int, nargs="+", default=[4266, 4267, 4268])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-recall", type=float, default=0.80)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = sorted(read_jsonl(args.dataset), key=lambda row: row["candidate_id"])
    dataset_rows = load_split(args.dataset)
    dataset_rows = sorted(dataset_rows, key=lambda row: row.candidate_id)
    assignments = {
        row["candidate_id"]: row
        for row in csv.DictReader(args.assignments.open(encoding="utf-8"))
    }
    labels = np.asarray([row.label for row in dataset_rows])
    families = [assignments[row.candidate_id]["task_family_id"] for row in dataset_rows]
    folds = [assignments[row.candidate_id]["validation_fold"] for row in dataset_rows]

    device = choose_device()
    print(f"Extracting frozen embeddings on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    encoder = AutoModel.from_pretrained(args.model_name, local_files_only=True).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    combined_mean = extract_embeddings(
        dataset_rows, tokenizer, encoder, device, args.batch_size, "mean"
    )
    fields = [field_texts(row["model_input"]) for row in raw_rows]
    request = extract_text_embeddings(
        tokenizer, encoder, device, [item[0] for item in fields], 128, args.batch_size
    )
    observation = extract_text_embeddings(
        tokenizer, encoder, device, [item[1] for item in fields], 256, args.batch_size
    )
    action = extract_text_embeddings(
        tokenizer, encoder, device, [item[2] for item in fields], 192, args.batch_size
    )
    del encoder
    if device.type == "mps":
        torch.mps.empty_cache()

    multi = torch.cat(
        (
            request,
            observation,
            action,
            torch.abs(request - action),
            request * action,
            torch.abs(observation - action),
            observation * action,
        ),
        dim=1,
    )
    rule_tensor = torch.tensor(
        [
            [float(policy_features(row["model_input"])[name]) for name in RULE_NAMES]
            for row in raw_rows
        ],
        dtype=torch.float32,
    )
    multi_rules = torch.cat((multi, rule_tensor), dim=1)
    configurations = [
        ("mean_full_weight", combined_mean, "full", False),
        ("mean_sqrt_weight", combined_mean, "sqrt", False),
        ("mean_no_weight", combined_mean, "none", False),
        ("multifield_no_weight", multi, "none", False),
        ("multifield_rules_no_weight", multi_rules, "none", False),
        ("multifield_rules_inner_threshold", multi_rules, "none", True),
    ]
    report = {
        "dataset": "public-workspace-509-v1",
        "protocol": "four-fold leave-task-families-out cross-validation",
        "encoder": args.model_name,
        "encoder_frozen": True,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "configurations": {},
    }
    prediction_rows = {
        row.candidate_id: {
            "candidate_id": row.candidate_id,
            "label": row.label,
            "fold": folds[index],
            "task_family_id": families[index],
        }
        for index, row in enumerate(dataset_rows)
    }
    for name, features, weight_mode, tune_threshold in configurations:
        print(f"Running {name}...")
        out_scores = np.zeros(len(labels), dtype=float)
        out_predictions = np.zeros(len(labels), dtype=int)
        fold_reports = []
        for fold in sorted(set(folds)):
            train_indices = [index for index, value in enumerate(folds) if value != fold]
            validation_indices = [index for index, value in enumerate(folds) if value == fold]
            if tune_threshold:
                threshold, threshold_selection = inner_threshold(
                    features,
                    labels,
                    train_indices,
                    families,
                    seeds=args.seeds,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_mode=weight_mode,
                    minimum_recall=args.minimum_recall,
                )
            else:
                threshold, threshold_selection = 0.5, None
            scores = fit_ensemble(
                features,
                labels,
                train_indices,
                validation_indices,
                seeds=args.seeds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_mode=weight_mode,
            )
            out_scores[validation_indices] = scores
            out_predictions[validation_indices] = (scores >= threshold).astype(int)
            fold_reports.append(
                {
                    "fold": fold,
                    "threshold": threshold,
                    "threshold_selection": threshold_selection,
                    "metrics": binary_metrics(labels[validation_indices], scores, threshold),
                }
            )
        # PR-AUC is threshold-free; decision metrics use each fold's leakage-free threshold.
        tn = int(((labels == 0) & (out_predictions == 0)).sum())
        fp = int(((labels == 0) & (out_predictions == 1)).sum())
        fn = int(((labels == 1) & (out_predictions == 0)).sum())
        tp = int(((labels == 1) & (out_predictions == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        from sklearn.metrics import average_precision_score

        aggregate = {
            "rows": len(labels),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "accuracy": (tn + tp) / len(labels),
            "unauthorized_precision": precision,
            "unauthorized_recall": recall,
            "unauthorized_f1": f1,
            "false_block_rate": fp / (fp + tn),
            "pr_auc": float(average_precision_score(labels, out_scores)),
        }
        report["configurations"][name] = {
            "feature_width": features.shape[1],
            "positive_weight": weight_mode,
            "threshold_policy": "inner task-family OOF" if tune_threshold else "fixed 0.5",
            "out_of_fold_micro": aggregate,
            "folds": fold_reports,
        }
        for index, row in enumerate(dataset_rows):
            prediction_rows[row.candidate_id][name] = {
                "probability_label_1": float(out_scores[index]),
                "prediction": int(out_predictions[index]),
            }

    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "out_of_fold_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for candidate_id in sorted(prediction_rows):
            handle.write(json.dumps(prediction_rows[candidate_id]) + "\n")
    print(
        json.dumps(
            {
                name: values["out_of_fold_micro"]
                for name, values in report["configurations"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
