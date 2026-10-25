import csv
import json
from collections import Counter
from pathlib import Path

batches = [
    {
        "name": "important_instructions/batch-002",
        "raw": Path("data/exports/important_instructions/batch-002/raw_calls.jsonl"),
        "csv": Path("data/annotations/annotation_sheet.csv"),
    },
    {
        "name": "tool_knowledge/batch-001",
        "raw": Path("data/exports/tool_knowledge/batch-001/raw_calls.jsonl"),
        "csv": Path("data/annotations/tool_knowledge_batch001/annotation_sheet.csv"),
    },
]

second_output = Path(
    "data/annotations/tool_knowledge_batch001/labeled_calls.jsonl"
)
combined_output = Path("data/annotations/labeled_calls_all.jsonl")
stats_output = Path("results/dataset_statistics_all.json")

all_calls = []
second_calls = []

for batch in batches:
    with batch["raw"].open(encoding="utf-8") as f:
        raw_calls = [
            json.loads(line) for line in f if line.strip()
        ]

    with batch["csv"].open(
        encoding="utf-8-sig", newline=""
    ) as f:
        annotations = list(csv.DictReader(f))

    if len(raw_calls) != len(annotations):
        raise ValueError(
            f"{batch['name']}: number of calls and labels does not match"
        )

    for i, (call, row) in enumerate(
        zip(raw_calls, annotations), start=1
    ):
        if int(row["row_no"]) != i:
            raise ValueError(
                f"{batch['name']}: row number mismatch at {i}"
            )

        if row["call_id"].strip() != str(call["call_id"]):
            raise ValueError(
                f"{batch['name']}: call ID mismatch at {i}"
            )

        label = row["label"].strip()
        injection = row["injection_present"].strip().upper()

        if label not in {"0", "1"}:
            raise ValueError(
                f"{batch['name']}: invalid label at row {i}"
            )

        if injection not in {"TRUE", "FALSE"}:
            raise ValueError(
                f"{batch['name']}: invalid injection at row {i}"
            )

        if not row["reason"].strip() or not row["evidence"].strip():
            raise ValueError(
                f"{batch['name']}: missing annotation at row {i}"
            )

        labeled = dict(call)

        labeled["label"] = int(label)
        labeled["injection_present"] = injection == "TRUE"

        labeled["annotation"] = {
            "reason": row["reason"].strip(),
            "evidence": row["evidence"].strip(),
            "review_status": row["review_status"].strip() or "pending",
            "reviewer": row["reviewer"].strip(),
        }

        # Identify the source batch in the combined dataset
        labeled["source_batch"] = batch["name"]

        all_calls.append(labeled)

        if batch["name"] == "tool_knowledge/batch-001":
            second_calls.append(labeled)

# Write files only after all validation has passed
second_output.parent.mkdir(parents=True, exist_ok=True)
combined_output.parent.mkdir(parents=True, exist_ok=True)
stats_output.parent.mkdir(parents=True, exist_ok=True)

def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

write_jsonl(second_output, second_calls)
write_jsonl(combined_output, all_calls)

statistics = {
    "total_calls": len(all_calls),
    "label_distribution": dict(
        Counter(str(c["label"]) for c in all_calls)
    ),
    "injection_present": sum(
        c["injection_present"] for c in all_calls
    ),
    "injection_absent": sum(
        not c["injection_present"] for c in all_calls
    ),
    "scenario_distribution": dict(
        Counter(c["scenario_type"] for c in all_calls)
    ),
    "tool_distribution": dict(
        Counter(c["tool_name"] for c in all_calls)
    ),
    "source_batch_distribution": dict(
        Counter(c["source_batch"] for c in all_calls)
    ),
    "review_status": dict(
        Counter(
            c["annotation"]["review_status"]
            for c in all_calls
        )
    ),
}

with stats_output.open("w", encoding="utf-8") as f:
    json.dump(statistics, f, indent=2, ensure_ascii=False)

print("\n===== EXPORT SUCCESSFUL =====")
print("Second batch:", len(second_calls))
print("Combined records:", len(all_calls))
print("Label distribution:", statistics["label_distribution"])
print("Injection present:", statistics["injection_present"])
print("Injection absent:", statistics["injection_absent"])
print("Review status:", statistics["review_status"])
print("\nSaved files:")
print(second_output)
print(combined_output)
print(stats_output)
