#!/usr/bin/env python3
"""Run a frozen DistilRoBERTa verifier over the grouped public-509 folds."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from tool_call_auditor.modeling import binary_metrics, load_split
from train_frozen_transformer import DenseVerifier, choose_device, extract_embeddings


def train_fixed_epochs(
    train_embeddings: torch.Tensor,
    train_labels: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> DenseVerifier:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = DenseVerifier(train_embeddings.shape[1])
    positives = int(train_labels.sum())
    negatives = len(train_labels) - positives
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    dataset = TensorDataset(
        train_embeddings, torch.tensor(train_labels, dtype=torch.float32)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(epochs):
        model.train()
        for embeddings, labels in loader:
            optimizer.zero_grad()
            loss = loss_function(model(embeddings), labels)
            loss.backward()
            optimizer.step()
    return model


def score(model: nn.Module, embeddings: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(embeddings)).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/splits/public_workspace_509_v1_cv")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/public_workspace_509_v1_cv/frozen_distilroberta"),
    )
    parser.add_argument("--model-name", default="distilbert/distilroberta-base")
    parser.add_argument("--seeds", type=int, nargs="+", default=[4266, 4267, 4268])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(path for path in args.data_dir.glob("fold_*") if path.is_dir())
    first_train = load_split(fold_dirs[0] / "train.jsonl")
    first_validation = load_split(fold_dirs[0] / "validation.jsonl")
    all_rows = sorted(first_train + first_validation, key=lambda row: row.candidate_id)
    if len(all_rows) != 509 or len({row.candidate_id for row in all_rows}) != 509:
        raise AssertionError("Expected 509 unique rows")

    device = choose_device()
    print(f"Extracting frozen embeddings on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    all_embeddings = extract_embeddings(
        all_rows, tokenizer, encoder, device, args.batch_size, "cls"
    )
    del encoder
    if device.type == "mps":
        torch.mps.empty_cache()
    index = {row.candidate_id: position for position, row in enumerate(all_rows)}
    assignments = {
        row["candidate_id"]: row
        for row in csv.DictReader(
            (args.data_dir / "fold_assignments.csv").open(encoding="utf-8")
        )
    }

    seed_reports = []
    seed_predictions = {}
    for seed in args.seeds:
        print(f"Training grouped dense heads for seed {seed}...")
        out_of_fold = {}
        fold_metrics = []
        for fold_dir in fold_dirs:
            train = load_split(fold_dir / "train.jsonl")
            validation = load_split(fold_dir / "validation.jsonl")
            train_indices = [index[row.candidate_id] for row in train]
            validation_indices = [index[row.candidate_id] for row in validation]
            model = train_fixed_epochs(
                all_embeddings[train_indices],
                np.asarray([row.label for row in train]),
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
            scores = score(model, all_embeddings[validation_indices])
            fold_metrics.append(
                {
                    "fold": fold_dir.name,
                    "metrics": binary_metrics(
                        [row.label for row in validation], scores, args.threshold
                    ),
                }
            )
            for row, probability in zip(validation, scores, strict=True):
                out_of_fold[row.candidate_id] = float(probability)
        ordered_scores = [out_of_fold[row.candidate_id] for row in all_rows]
        seed_predictions[seed] = ordered_scores
        seed_reports.append(
            {
                "seed": seed,
                "out_of_fold_micro": binary_metrics(
                    [row.label for row in all_rows], ordered_scores, args.threshold
                ),
                "folds": fold_metrics,
            }
        )

    ensemble_scores = np.mean(
        np.asarray([seed_predictions[seed] for seed in args.seeds]), axis=0
    )
    labels = [row.label for row in all_rows]
    overall = binary_metrics(labels, ensemble_scores, args.threshold)
    source_metrics = {}
    for source in sorted({row["source_dataset"] for row in assignments.values()}):
        mask = [assignments[row.candidate_id]["source_dataset"] == source for row in all_rows]
        source_metrics[source] = binary_metrics(
            [label for label, keep in zip(labels, mask, strict=True) if keep],
            [score for score, keep in zip(ensemble_scores, mask, strict=True) if keep],
            args.threshold,
        )
    report = {
        "protocol": "four-fold leave-task-families-out cross-validation",
        "model_name": args.model_name,
        "encoder_frozen": True,
        "pooling": "cls",
        "head": "768 -> 64 -> 1",
        "epochs_fixed_before_cv": args.epochs,
        "threshold": args.threshold,
        "seeds": seed_reports,
        "three_seed_probability_ensemble": overall,
        "per_source": source_metrics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "out_of_fold_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for position, row in enumerate(all_rows):
            handle.write(
                json.dumps(
                    {
                        "candidate_id": row.candidate_id,
                        "source_dataset": assignments[row.candidate_id]["source_dataset"],
                        "fold": assignments[row.candidate_id]["validation_fold"],
                        "task_family_id": assignments[row.candidate_id]["task_family_id"],
                        "label": row.label,
                        "probability_label_1": float(ensemble_scores[position]),
                        "prediction": int(ensemble_scores[position] >= args.threshold),
                        "seed_scores": {
                            str(seed): seed_predictions[seed][position] for seed in args.seeds
                        },
                    }
                )
                + "\n"
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
