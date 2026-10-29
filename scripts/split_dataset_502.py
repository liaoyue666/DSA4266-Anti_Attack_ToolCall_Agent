from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/annotations/dataset_502_v1/labeled_calls.jsonl"
DEFAULT_FAMILIES = ROOT / "data/manifests/dataset_502_v1_task_families.json"
DEFAULT_OUTPUT = ROOT / "data/splits/dataset_502_v1"
SPLITS = ("train", "validation", "test")


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass(frozen=True)
class ComponentStats:
    groups: tuple[str, ...]
    row_count: int
    positive_count: int
    tools: Counter[str]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _validate_records(records: list[dict]) -> None:
    if not records:
        raise ValueError("The labeled dataset is empty.")
    candidate_ids = [row["candidate_id"] for row in records]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_id values must be unique before splitting.")
    invalid_labels = sorted({row.get("label") for row in records} - {0, 1})
    if invalid_labels:
        raise ValueError(f"All rows must have binary labels; found {invalid_labels!r}.")
    for row in records:
        model_input = row.get("model_input")
        if not isinstance(model_input, dict):
            raise ValueError(f"{row['candidate_id']} has no model_input object.")
        expected = {"original_user_request", "observations_before_call", "proposed_action"}
        if set(model_input) != expected:
            raise ValueError(
                f"{row['candidate_id']} model_input keys differ from {sorted(expected)}."
            )


def _load_declared_families(path: Path) -> tuple[list[dict], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    families = payload.get("families", [])
    group_to_family: dict[str, str] = {}
    for family in families:
        family_id = family["family_id"]
        for group in family["task_group_ids"]:
            previous = group_to_family.setdefault(group, family_id)
            if previous != family_id:
                raise ValueError(f"{group} appears in multiple declared task families.")
    return families, group_to_family


def _build_components(
    records: list[dict], family_path: Path
) -> tuple[list[tuple[str, ...]], dict[str, str]]:
    groups = sorted({row["task_group_id"] for row in records})
    known_groups = set(groups)
    union_find = UnionFind(groups)

    families, declared_family = _load_declared_families(family_path)
    for family in families:
        present = [group for group in family["task_group_ids"] if group in known_groups]
        for group in present[1:]:
            union_find.union(present[0], group)

    for row in records:
        current = row["task_group_id"]
        for related in row.get("related_task_group_ids", []):
            if related in known_groups:
                union_find.union(current, related)

    for field in ("model_input_fingerprint", "near_duplicate_group_id"):
        first_group: dict[str, str] = {}
        for row in records:
            value = row.get(field)
            if not value:
                continue
            current = row["task_group_id"]
            if value in first_group:
                union_find.union(current, first_group[value])
            else:
                first_group[value] = current

    component_groups: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        component_groups[union_find.find(group)].append(group)
    components = sorted(tuple(sorted(values)) for values in component_groups.values())

    group_to_family: dict[str, str] = {}
    for component in components:
        declared_ids = {declared_family[group] for group in component if group in declared_family}
        if len(declared_ids) > 1:
            raise ValueError(f"Linked component joins multiple declared families: {component}")
        family_id = next(iter(declared_ids), component[0])
        for group in component:
            group_to_family[group] = family_id
    return components, group_to_family


def _component_stats(
    components: list[tuple[str, ...]], records_by_group: dict[str, list[dict]]
) -> list[ComponentStats]:
    result = []
    for component in components:
        rows = [row for group in component for row in records_by_group[group]]
        result.append(
            ComponentStats(
                groups=component,
                row_count=len(rows),
                positive_count=sum(row["label"] == 1 for row in rows),
                tools=Counter(
                    row["model_input"]["proposed_action"]["tool_name"] for row in rows
                ),
            )
        )
    return result


def _sum_stats(
    indexes: Iterable[int], stats: list[ComponentStats]
) -> tuple[int, int, Counter[str]]:
    row_count = 0
    positive_count = 0
    tools: Counter[str] = Counter()
    for index in indexes:
        item = stats[index]
        row_count += item.row_count
        positive_count += item.positive_count
        tools.update(item.tools)
    return row_count, positive_count, tools


def _split_score(
    validation: tuple[int, ...],
    test: tuple[int, ...],
    stats: list[ComponentStats],
    total_rows: int,
    total_positives: int,
    all_tools: Counter[str],
) -> float:
    all_indexes = set(range(len(stats)))
    train = all_indexes - set(validation) - set(test)
    score = 0.0
    for indexes, target in ((train, 0.70), (validation, 0.15), (test, 0.15)):
        rows, positives, tools = _sum_stats(indexes, stats)
        score += 6 * ((rows / total_rows - target) / target) ** 2
        score += 6 * ((positives / total_positives - target) / target) ** 2
        for tool, count in all_tools.items():
            expected = count * target
            score += 0.2 * ((tools[tool] - expected) / max(expected, 3)) ** 2

    for indexes in (validation, test):
        _, positives, tools = _sum_stats(indexes, stats)
        if positives == 0:
            score += 1000
        if len(tools) < 2:
            score += 20

    _, _, train_tools = _sum_stats(train, stats)
    for tool in all_tools:
        if train_tools[tool] == 0:
            score += 10000
    return score


def _choose_components(
    stats: list[ComponentStats], records: list[dict], seed: int, trials: int
) -> dict[str, set[int]]:
    if len(stats) < 7:
        raise ValueError("At least seven independent task components are required.")

    all_tools = Counter(
        row["model_input"]["proposed_action"]["tool_name"] for row in records
    )
    component_tool_coverage = Counter()
    for item in stats:
        component_tool_coverage.update(item.tools.keys())
    pinned_to_train = {
        index
        for index, item in enumerate(stats)
        if any(component_tool_coverage[tool] == 1 for tool in item.tools)
    }
    eligible = [index for index in range(len(stats)) if index not in pinned_to_train]

    validation_count = max(1, round(len(stats) * 0.15))
    test_count = max(1, round(len(stats) * 0.15))
    if len(eligible) < validation_count + test_count:
        raise ValueError("Too many components are pinned to train to create validation/test.")

    total_positives = sum(row["label"] == 1 for row in records)
    random_generator = random.Random(seed)
    best: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
    for _ in range(trials):
        shuffled = eligible.copy()
        random_generator.shuffle(shuffled)
        validation = tuple(sorted(shuffled[:validation_count]))
        test = tuple(sorted(shuffled[validation_count : validation_count + test_count]))
        score = _split_score(
            validation,
            test,
            stats,
            len(records),
            total_positives,
            all_tools,
        )
        if best is None or (score, validation, test) < best:
            best = (score, validation, test)

    assert best is not None
    validation = set(best[1])
    test = set(best[2])
    train = set(range(len(stats))) - validation - test
    return {"train": train, "validation": validation, "test": test}


def _stats_for_rows(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "labels": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        "source_types": dict(sorted(Counter(row["source_type"] for row in rows).items())),
        "tools": dict(
            sorted(
                Counter(
                    row["model_input"]["proposed_action"]["tool_name"] for row in rows
                ).items()
            )
        ),
        "task_groups": sorted({row["task_group_id"] for row in rows}),
    }


def build_split(
    input_path: Path,
    family_path: Path,
    output_dir: Path,
    seed: int = 4266,
    trials: int = 100_000,
) -> dict:
    records = _read_jsonl(input_path)
    _validate_records(records)
    records_by_group: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        records_by_group[row["task_group_id"]].append(row)

    components, group_to_family = _build_components(records, family_path)
    stats = _component_stats(components, records_by_group)
    selected = _choose_components(stats, records, seed, trials)

    group_to_split: dict[str, str] = {}
    for split, indexes in selected.items():
        for index in indexes:
            for group in stats[index].groups:
                group_to_split[group] = split

    split_rows: dict[str, list[dict]] = {split: [] for split in SPLITS}
    for row in records:
        split_rows[group_to_split[row["task_group_id"]]].append(row)
    for rows in split_rows.values():
        rows.sort(key=lambda row: row["candidate_id"])

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        training_rows = [
            {
                "candidate_id": row["candidate_id"],
                "label": row["label"],
                "model_input": row["model_input"],
            }
            for row in split_rows[split]
        ]
        _write_jsonl(output_dir / f"{split}.jsonl", training_rows)

    manifest_fields = [
        "candidate_id",
        "split",
        "task_group_id",
        "task_family_id",
        "label",
        "source_type",
        "tool_name",
        "model_input_fingerprint",
        "near_duplicate_group_id",
    ]
    with (output_dir / "split_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for split in SPLITS:
            for row in split_rows[split]:
                writer.writerow(
                    {
                        "candidate_id": row["candidate_id"],
                        "split": split,
                        "task_group_id": row["task_group_id"],
                        "task_family_id": group_to_family[row["task_group_id"]],
                        "label": row["label"],
                        "source_type": row["source_type"],
                        "tool_name": row["model_input"]["proposed_action"]["tool_name"],
                        "model_input_fingerprint": row["model_input_fingerprint"],
                        "near_duplicate_group_id": row.get("near_duplicate_group_id") or "",
                    }
                )

    with (output_dir / "task_group_assignments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task_group_id", "task_family_id", "split", "row_count"],
        )
        writer.writeheader()
        for group in sorted(records_by_group):
            writer.writerow(
                {
                    "task_group_id": group,
                    "task_family_id": group_to_family[group],
                    "split": group_to_split[group],
                    "row_count": len(records_by_group[group]),
                }
            )

    statistics = {
        "dataset": "dataset_502_v1",
        "seed": seed,
        "search_trials": trials,
        "source_path": str(input_path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "task_family_path": str(family_path.relative_to(ROOT)),
        "total_rows": len(records),
        "split_statistics": {
            split: _stats_for_rows(split_rows[split]) for split in SPLITS
        },
    }
    (output_dir / "split_statistics.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return statistics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leakage-aware train/validation/test files for dataset-502-v1."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--task-families", type=Path, default=DEFAULT_FAMILIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=4266)
    parser.add_argument("--trials", type=int, default=100_000)
    args = parser.parse_args()
    statistics = build_split(
        args.input.resolve(),
        args.task_families.resolve(),
        args.output_dir.resolve(),
        args.seed,
        args.trials,
    )
    print(json.dumps(statistics["split_statistics"], indent=2))


if __name__ == "__main__":
    main()
