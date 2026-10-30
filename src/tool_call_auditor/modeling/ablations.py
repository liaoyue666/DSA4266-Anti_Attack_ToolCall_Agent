"""Label-blind context ablations shared by sparse and frozen models."""

from copy import deepcopy
from dataclasses import replace

from .data import DatasetRow

ABLATIONS = ("full", "no_request", "no_observations", "action_only")


def ablate_input(model_input: dict, mode: str) -> dict:
    if mode not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {mode}")
    result = deepcopy(model_input)
    if mode in ("no_request", "action_only"):
        result["original_user_request"] = ""
    if mode in ("no_observations", "action_only"):
        result["observations_before_call"] = []
    return result


def ablate_rows(rows: list[DatasetRow], mode: str) -> list[DatasetRow]:
    return [replace(row, model_input=ablate_input(row.model_input, mode)) for row in rows]
