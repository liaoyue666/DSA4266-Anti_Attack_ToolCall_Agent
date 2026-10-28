import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

source = "origin/main:data/delivery/dataset-502-v1/annotation_candidates.jsonl"

output = Path(
    "data/annotations/dataset_502_v1/annotation_sheet.csv"
)

# Protect any existing manual annotations
if output.exists():
    raise SystemExit(
        "Annotation CSV already exists. Refusing to overwrite it."
    )

# Read the new dataset directly from origin/main
result = subprocess.check_output(
    ["git", "show", source],
    text=True
)

candidates = [
    json.loads(line)
    for line in result.splitlines()
    if line.strip()
]

# Prevent spreadsheet software from interpreting
# untrusted text as formulas
def safe_cell(value):
    text = str(value)
    if text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text

columns = [
    "row_no",
    "candidate_id",
    "source_type",
    "task_group_id",
    "original_user_request",
    "tool_name",
    "tool_arguments",
    "observations_preview",
    "label",
    "injection_present",
    "reason",
    "evidence",
    "review_status",
    "reviewer"
]

with output.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=columns
    )

    writer.writeheader()

    for i, candidate in enumerate(candidates, 1):

        model_input = candidate["model_input"]
        action = model_input["proposed_action"]

        observations = model_input.get(
            "observations_before_call", []
        )

        preview = json.dumps(
            observations,
            ensure_ascii=False
        )[:600]

        writer.writerow({
            "row_no": i,
            "candidate_id": candidate["candidate_id"],
            "source_type": candidate["source_type"],
            "task_group_id": candidate.get(
                "task_group_id", ""
            ),
            "original_user_request": safe_cell(
                model_input["original_user_request"]
            ),
            "tool_name": action["tool_name"],
            "tool_arguments": safe_cell(
                json.dumps(
                    action["arguments"],
                    ensure_ascii=False
                )
            ),
            "observations_preview": safe_cell(preview),
            "label": "",
            "injection_present": "",
            "reason": "",
            "evidence": "",
            "review_status": "pending",
            "reviewer": ""
        })

print("\n===== ANNOTATION SHEET CREATED =====")
print("Total candidates:", len(candidates))
print(
    "Source distribution:",
    dict(Counter(
        c["source_type"] for c in candidates
    ))
)
print("Saved to:", output)
