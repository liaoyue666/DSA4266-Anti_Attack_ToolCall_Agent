import csv
import json
from collections import Counter
from pathlib import Path

raw_path = Path("data/exports/important_instructions/batch-002/raw_calls.jsonl")
csv_path = Path("data/annotations/annotation_sheet.csv")
output_path = Path("data/annotations/labeled_calls.jsonl")
stats_path = Path("results/dataset_statistics.json")

# Read original calls
with open(raw_path, encoding="utf-8") as f:
    raw_calls = [json.loads(line) for line in f if line.strip()]

# Read manual labels
with open(csv_path, encoding="utf-8-sig", newline="") as f:
    annotations = list(csv.DictReader(f))

if len(raw_calls) != len(annotations):
    raise ValueError("Number of calls and annotations does not match.")

labeled_calls = []

for i, (call, row) in enumerate(zip(raw_calls, annotations), start=1):

    if int(row["row_no"]) != i:
        raise ValueError(f"Row number mismatch at row {i}")

    if row["call_id"].strip() != str(call["call_id"]):
        raise ValueError(f"Call ID mismatch at row {i}")

    label = row["label"].strip()
    injection = row["injection_present"].strip().upper()

    if label not in ["0", "1"]:
        raise ValueError(f"Invalid label at row {i}")

    if injection not in ["TRUE", "FALSE"]:
        raise ValueError(f"Invalid injection flag at row {i}")

    if not row["reason"].strip() or not row["evidence"].strip():
        raise ValueError(f"Missing reason or evidence at row {i}")

    labeled = dict(call)
    labeled["label"] = int(label)
    labeled["injection_present"] = injection == "TRUE"

    labeled["annotation"] = {
        "reason": row["reason"].strip(),
        "evidence": row["evidence"].strip(),
        "review_status": row["review_status"].strip() or "pending",
        "reviewer": row["reviewer"].strip()
    }

    labeled_calls.append(labeled)

# Export labeled JSONL
with open(output_path, "w", encoding="utf-8") as f:
    for call in labeled_calls:
        f.write(json.dumps(call, ensure_ascii=False) + "\n")

# Generate statistics
statistics = {
    "total_calls": len(labeled_calls),
    "label_distribution": dict(
        Counter(str(c["label"]) for c in labeled_calls)
    ),
    "injection_present": sum(
        c["injection_present"] for c in labeled_calls
    ),
    "injection_absent": sum(
        not c["injection_present"] for c in labeled_calls
    ),
    "scenario_distribution": dict(
        Counter(c["scenario_type"] for c in labeled_calls)
    ),
    "tool_distribution": dict(
        Counter(c["tool_name"] for c in labeled_calls)
    ),
    "review_status": dict(
        Counter(c["annotation"]["review_status"] for c in labeled_calls)
    )
}

with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(statistics, f, indent=2, ensure_ascii=False)

print("\nExport successful!")
print("Total labeled calls:", len(labeled_calls))
print("Label distribution:", statistics["label_distribution"])
print("Injection present:", statistics["injection_present"])
print("Review status:", statistics["review_status"])
print("Saved to:", output_path)
print("Statistics saved to:", stats_path)
