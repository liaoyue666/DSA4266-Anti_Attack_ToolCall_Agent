from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_workspace509_merge_and_grouped_folds(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    split_dir = tmp_path / "splits"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_public_workspace509_v1.py"),
            "--dataset-dir",
            str(dataset_dir),
            "--split-dir",
            str(split_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [json.loads(line) for line in (dataset_dir / "dataset.jsonl").read_text().splitlines()]
    assert len(rows) == 509
    assert len({row["candidate_id"] for row in rows}) == 509
    assert {row["source_dataset"] for row in rows} == {
        "public_workspace_120_v1",
        "public_workspace_400_v1_confirmed",
    }
    assignments = list(csv.DictReader((split_dir / "fold_assignments.csv").open()))
    assert len(assignments) == 509
    family_folds = {}
    for row in assignments:
        family_folds.setdefault(row["task_family_id"], set()).add(row["validation_fold"])
    assert all(len(folds) == 1 for folds in family_folds.values())
    stats = json.loads((split_dir / "cross_validation_statistics.json").read_text())
    assert stats["overall"]["labels"] == {"0": 400, "1": 109}
    assert sum(fold["validation"]["rows"] for fold in stats["folds"].values()) == 509
    assert all(set(fold["validation"]["labels"]) == {"0", "1"} for fold in stats["folds"].values())
