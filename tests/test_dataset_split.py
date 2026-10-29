from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data/splits/dataset_502_v1"


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _manifest() -> list[dict]:
    with (SPLIT_DIR / "split_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle))


def test_training_files_cover_every_candidate_once_without_metadata_leakage() -> None:
    rows_by_split = {
        split: _jsonl(SPLIT_DIR / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    assert len(all_rows) == 502
    assert len({row["candidate_id"] for row in all_rows}) == 502
    for rows in rows_by_split.values():
        assert {row["label"] for row in rows} == {0, 1}
        for row in rows:
            assert set(row) == {"candidate_id", "label", "model_input"}
            assert set(row["model_input"]) == {
                "original_user_request",
                "observations_before_call",
                "proposed_action",
            }


def test_task_families_and_duplicate_keys_do_not_cross_splits() -> None:
    rows = _manifest()
    for field in (
        "task_group_id",
        "task_family_id",
        "model_input_fingerprint",
        "near_duplicate_group_id",
    ):
        splits_by_value: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row[field]:
                splits_by_value[row[field]].add(row["split"])
        assert all(len(splits) == 1 for splits in splits_by_value.values())


def test_train_contains_every_tool_and_split_statistics_match_files() -> None:
    train = _jsonl(SPLIT_DIR / "train.jsonl")
    train_tools = {
        row["model_input"]["proposed_action"]["tool_name"] for row in train
    }
    assert train_tools == {
        "add_calendar_event_participants",
        "append_to_file",
        "create_calendar_event",
        "create_file",
        "delete_file",
        "reschedule_calendar_event",
        "send_email",
        "share_file",
    }

    statistics = json.loads(
        (SPLIT_DIR / "split_statistics.json").read_text(encoding="utf-8")
    )
    assert statistics["total_rows"] == 502
    for split in ("train", "validation", "test"):
        rows = _jsonl(SPLIT_DIR / f"{split}.jsonl")
        assert statistics["split_statistics"][split]["rows"] == len(rows)
