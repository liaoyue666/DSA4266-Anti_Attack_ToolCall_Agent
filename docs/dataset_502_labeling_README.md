# Member B — Dataset 502 Provisional Annotations (v1)

## Scope and status

This package provides **AI-assisted provisional annotations** for all 502 candidates in `data/delivery/dataset-502-v1/annotation_candidates.jsonl`, based on the matching `review_metadata.jsonl` for constructed candidates. It has **not** undergone independent human adjudication. In the annotation sheet, `review_status=pending` and `reviewer` is intentionally empty for **every row**.

| Source type | Count | Provisional Label 0 | Provisional Label 1 |
|---|---:|---:|---:|
| Observed: agent-proposed calls | 382 | 382 | 0 |
| Constructed: counterfactual candidates | 120 | 0 | 120 |
| Total | 502 | 382 | 120 |

The observed records include 351 attack-context calls where an injection marker is present in the pre-call observations, and 31 normal-context calls. **An injected observation is not itself an unauthorized tool call.** The observed labels are provisional task-specific checks of the action versus the user request, not a claim that all attacks failed or that a real-episode ASR is zero. The 120 constructed candidates have not been proposed/executed by an agent and **must not** count toward real-episode ASR.

### Deliverables

- `data/annotations/dataset_502_v1/annotation_sheet.csv`: authoritative **502-row working annotation sheet**. Same 16-column layout/order as the earlier `annotation_sheet.csv` (`J=observation_preview`, `K=label`, `L=injection_present`, `M=reason`, `N=evidence`, `O=review_status`, `P=reviewer`). UTF-8 with BOM for WPS.
- `data/annotations/dataset_502_v1/observed_382_annotation_sheet.csv`: 382-row convenience filtered view; the 502-row sheet is the master.
- `data/annotations/dataset_502_v1/constructed_120_annotation_sheet.csv`: 120-row convenience filtered view.
- `data/annotations/dataset_502_v1/labeled_calls.jsonl`: all 502 source candidate objects preserving the **exact original `model_input`** and adding a top-level preliminary `label`, `injection_present`, and an `annotation` object.
- `data/annotations/dataset_502_v1/annotation_audit.csv`: maps row numbers back to `candidate_id`, source type, `model_input_fingerprint`, `task_group_id`, `related_task_group_ids`, and annotation origin/review priority.
- `data/annotations/dataset_502_v1/review_queue.csv`: 134 flagged records for focused human adjudication: 120 synthetic counterfactuals and 14 observed calls with optional/ambiguous fields. **All 502 remain pending review**, not only these 134.
- `results/dataset_502_labeling_statistics.json`: counts and limitations.
- `results/source_sha256.json`: SHA-256 checksums for the two input JSONL files used to prepare this delivery.
- `scripts/validate_dataset_502_delivery.py`: run this before commit/push.
- `scripts/export_dataset_502_labels.py`: validates annotations and regenerates JSONL, split CSVs, and statistics after later human changes to the master CSV.

### Column semantics and provenance

The 16-column master sheet follows the original Member B pilot schema:

`row_no, call_id, episode_id, user_task_id, scenario_type, injection_task_id, original_user_request, tool_name, tool_arguments, observation_preview, label, injection_present, reason, evidence, review_status, reviewer`

- For **observed** candidates, `call_id`, `episode_id`, and task information come from the first preserved `parent_sources` reference. Additional parents remain in JSONL.
- For **constructed** candidates, the B-column `call_id` stores **the constructed `candidate_id` as an ID placeholder**, NOT an actual executed call ID. `episode_id` is left blank, and `scenario_type=constructed`.
- `observation_preview` contains only the first 1,000 characters for WPS readability. **A reviewer must consult the full `observations_before_call` from the JSONL** before confirming a label.
- The `reason` / `evidence` fields are English, case-specific, preliminary explanations of the binary authorization decision. Review metadata was used to validate constructed mutations, not put into `model_input`.
- Normal observed calls have `injection_present=FALSE`. All 351 attack-context observed calls had an `<INFORMATION>` marker in the source pre-call observations and have `injection_present=TRUE`; all 120 constructed candidates also have `injection_present=TRUE`.
- The `label` is an **authorization judgement for the proposed action**, not a score for attack presence or an execution/success record. Label 0 = authorized; Label 1 = unauthorized.

### Human-review priority

Focus first on `review_queue.csv`, then independently review other records (e.g. stratified by tool and task group). The 14 flagged observed records comprise:

- 7 family reunion events that include the organizer as a participant even though an invitation was not explicitly requested;
- 4 follow-up meetings that copy the earlier meeting location despite a user instruction referring only to the same participants;
- 1 lunch event with an optional `location` field;
- 1 dental reschedule with a `null` new end-time;
- 1 team meeting TODO email containing an additional attachment not requested by the user (**high priority**: review whether sending the file exceeds recipient authorization).

The preliminary Label 0 for these 14 observations should be treated as *particularly uncertain*, and may change after adjudication. All 120 constructed records also require independent human review; `expected_unauthorized` is only a construction intent, not ground truth.

### How to use this branch

Copy the files from the ZIP into the repo **root** (do not copy the enclosing ZIP filename as a folder). You need not merge `main` or duplicate A's raw 502-row source delivery on your feature branch. The verification/export scripts use the local `data/delivery/dataset-502-v1/annotation_candidates.jsonl` when present, otherwise `git show origin/main:...` after fetching `origin/main`.

```bash
cd ~/Desktop/DSA4266-Anti_Attack_ToolCall_Agent
git branch --show-current      # MUST say wjq/data-labeling
git fetch origin main
python3 scripts/validate_dataset_502_delivery.py

git add data/annotations/dataset_502_v1/ docs/dataset_502_labeling_README.md scripts/export_dataset_502_labels.py scripts/validate_dataset_502_delivery.py
git add -f results/dataset_502_labeling_statistics.json results/source_sha256.json
git diff --cached --stat
git commit -m "Add provisional labels and QC for dataset 502"
git push origin wjq/data-labeling
```

Do **not** overwrite the prior pilot `data/annotations/annotation_sheet.csv` or the source files under `data/delivery/dataset-502-v1/`.

### How to update after team review

Edit **only the 502-row master CSV** in WPS, preserving the 16 column names and order, source IDs, original requests, and tool arguments. Change `label` or `injection_present` only when supported by the full input context. Update the English `reason` and `evidence` if the label changes. Set `review_status=reviewed` and supply a real `reviewer` only after that person actually reviews the case; use `disputed` for unresolved disagreements. Keep `pending` until then.

Regenerate the associated JSONL/statistics and verify consistency:

```bash
python3 scripts/export_dataset_502_labels.py
python3 scripts/validate_dataset_502_delivery.py
```

Recheck/resolve any review queue items manually, and update the queue as needed. `review_queue.csv` is an **initial triage queue**, not a substitute for the complete set of pending statuses.

### Train/validation/test leakage restrictions

Every candidate with the same `task_group_id` belongs to one split. If an observed deduplicated candidate lists multiple `related_task_group_ids`, those task groups are linked and must be allocated to the **same split**. Every constructed child must remain with its parent task group. The team must agree on the held-out evaluation split.

The only classifier-visible input is `model_input`:

- `original_user_request`
- `observations_before_call`
- `proposed_action` (`tool_name`, `arguments`)

Do **not** feed `label`, `source_type`, `task_group_id`, review reasons, reviewer status, `review_metadata.jsonl`, construction methods, or related IDs to a predictive model. These remain outside `model_input` solely for audit/splitting/review.

### Limitations

- Independent expert/human validation has not occurred. A Git push of these files completes an **annotation draft delivery**, not gold-label adjudication.
- 502 entries are not independent user tasks; there are repeated and related task groups.
- Binary labels do not measure action execution success, effectiveness of prompt injection, or model generalization.
