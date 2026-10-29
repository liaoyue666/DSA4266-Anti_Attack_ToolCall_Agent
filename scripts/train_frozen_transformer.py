#!/usr/bin/env python3
"""Train a small dense verifier head over frozen DistilRoBERTa embeddings."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from tool_call_auditor.modeling import binary_metrics, load_split, select_threshold
from tool_call_auditor.modeling.transformer_input import encode_model_input


class DenseVerifier(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.layers(embeddings).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/splits/dataset_502_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/frozen_distilroberta"))
    parser.add_argument("--model-name", default="distilbert/distilroberta-base")
    parser.add_argument("--seeds", type=int, nargs="+", default=[4266, 4267, 4268])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--minimum-recall", type=float, default=0.80)
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def extract_embeddings(rows, tokenizer, encoder, device, batch_size: int, pooling: str) -> torch.Tensor:
    encoded = [encode_model_input(tokenizer, row.model_input) for row in rows]
    input_ids = torch.tensor([item["input_ids"] for item in encoded], dtype=torch.long)
    attention_mask = torch.tensor([item["attention_mask"] for item in encoded], dtype=torch.long)
    dataset = TensorDataset(input_ids, attention_mask)
    embeddings = []
    encoder.eval()
    with torch.inference_mode():
        for batch_ids, batch_mask in DataLoader(dataset, batch_size=batch_size):
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            output = encoder(input_ids=batch_ids, attention_mask=batch_mask)
            if pooling == "cls":
                pooled = output.last_hidden_state[:, 0]
            else:
                pooled = mean_pool(output.last_hidden_state, batch_mask)
            embeddings.append(pooled.cpu())
    return torch.cat(embeddings)


def probabilities(model: nn.Module, embeddings: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(embeddings)).cpu().numpy()


def train_seed(args, seed: int, embeddings, labels) -> tuple[DenseVerifier, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = DenseVerifier(embeddings["train"].shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    positive = int(labels["train"].sum())
    negative = len(labels["train"]) - positive
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive))
    train_dataset = TensorDataset(
        embeddings["train"], torch.tensor(labels["train"], dtype=torch.float32)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

    best_state = None
    best_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    validation_targets = torch.tensor(labels["validation"], dtype=torch.float32)
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_embeddings, batch_labels in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_embeddings), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_labels)
        model.eval()
        with torch.inference_mode():
            validation_loss = loss_function(
                model(embeddings["validation"]), validation_targets
            ).item()
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_dataset),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"best_validation_loss": best_loss, "history": history}


def write_predictions(path: Path, rows, scores: np.ndarray, threshold: float) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row, score in zip(rows, scores, strict=True):
            handle.write(
                json.dumps(
                    {
                        "candidate_id": row.candidate_id,
                        "label": row.label,
                        "probability_label_1": float(score),
                        "prediction": int(score >= threshold),
                    }
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    print(f"Extracting frozen embeddings on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    splits = {
        name: load_split(args.data_dir / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    labels = {name: np.asarray([row.label for row in rows]) for name, rows in splits.items()}
    embeddings = {
        name: extract_embeddings(rows, tokenizer, encoder, device, args.batch_size, args.pooling)
        for name, rows in splits.items()
    }
    del encoder
    if device.type == "mps":
        torch.mps.empty_cache()

    runs = []
    for seed in args.seeds:
        print(f"Training dense head with seed {seed}...")
        model, training = train_seed(args, seed, embeddings, labels)
        scores = {name: probabilities(model, value) for name, value in embeddings.items()}
        threshold, selection = select_threshold(
            labels["validation"], scores["validation"], minimum_recall=args.minimum_recall
        )
        metrics = {
            name: binary_metrics(labels[name], scores[name], threshold) for name in splits
        }
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "hidden_size": embeddings["train"].shape[1],
                "model_name": args.model_name,
                "threshold": threshold,
                "input_budgets": {"request": 96, "observation": 256, "action": 156},
            },
            seed_dir / "dense_head.pt",
        )
        for name, rows in splits.items():
            write_predictions(seed_dir / f"{name}_predictions.jsonl", rows, scores[name], threshold)
        run = {
            "seed": seed,
            "threshold": threshold,
            "validation_selection": selection,
            "training": training,
            "metrics": metrics,
        }
        (seed_dir / "metrics.json").write_text(json.dumps(run, indent=2) + "\n")
        runs.append(run)

    summary = {
        "model_name": args.model_name,
        "encoder_frozen": True,
        "pooling": args.pooling,
        "device_used_for_embedding_extraction": str(device),
        "seeds": args.seeds,
        "runs": runs,
        "test_mean": {
            metric: float(np.mean([run["metrics"]["test"][metric] for run in runs]))
            for metric in (
                "unauthorized_precision",
                "unauthorized_recall",
                "unauthorized_f1",
                "false_block_rate",
                "pr_auc",
            )
        },
        "test_std": {
            metric: float(np.std([run["metrics"]["test"][metric] for run in runs]))
            for metric in (
                "unauthorized_precision",
                "unauthorized_recall",
                "unauthorized_f1",
                "false_block_rate",
                "pr_auc",
            )
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["test_mean"], indent=2))


if __name__ == "__main__":
    main()
