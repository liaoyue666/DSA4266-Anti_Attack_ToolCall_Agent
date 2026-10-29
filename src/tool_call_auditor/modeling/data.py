"""Loading for the leakage-aware dataset-502-v1 split."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetRow:
    candidate_id: str
    label: int
    model_input: dict[str, Any]


def load_split(path: str | Path) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            label = raw.get("label")
            if label not in (0, 1):
                raise ValueError(f"{path}:{line_number}: expected binary label")
            model_input = raw.get("model_input")
            if not isinstance(model_input, dict):
                raise ValueError(f"{path}:{line_number}: model_input is not an object")
            rows.append(
                DatasetRow(
                    candidate_id=str(raw["candidate_id"]),
                    label=label,
                    model_input=model_input,
                )
            )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows
