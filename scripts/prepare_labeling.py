import json
import csv
from pathlib import Path

# A's original data (read-only)
INPUT = Path(
    "data/exports/important_instructions/"
    "batch-002/raw_calls.jsonl"
)

# B's annotation sheet
OUTPUT = Path("data/annotations/annotation_sheet.csv")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(INPUT, "r", encoding="utf-8") as f:
    calls = [json.loads(line) for line in f if line.strip()]

rows = []

for i, call in enumerate(calls, start=1):
    observations = call.get("observation_provenance") or []

    observation_text = "\n\n".join(
        str(obs.get("content", ""))
        for obs in observations
        if isinstance(obs, dict)
    )

    rows.append({
        "row_no": i,
        "call_id": call["call_id"],
        "episode_id": call["episode_id"],
        "user_task_id": call.get("user_task_id", ""),
        "scenario_type": call.get("scenario_type", ""),
        "injection_task_id": call.get("injection_task_id", ""),
        "original_user_request": call.get(
            "original_user_request", ""
        ),
        "tool_name": call.get("tool_name", ""),
        "tool_arguments": json.dumps(
            call.get("tool_arguments", {}),
            ensure_ascii=False
        ),
        "observation_preview": observation_text[:1000],
        "label": "",
        "injection_present": "",
        "reason": "",
        "evidence": "",
        "review_status": "pending",
        "reviewer": ""
    })

if not rows:
    raise ValueError("No tool calls found.")

with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)

print("CSV created successfully!")
print("Total calls:", len(rows))
print("Saved to:", OUTPUT)
