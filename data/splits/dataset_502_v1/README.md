# dataset-502-v1 train/validation/test split

This directory contains a deterministic, leakage-aware split of the 502 labeled
tool-call candidates in `data/annotations/dataset_502_v1/labeled_calls.jsonl`.

| Split | Rows | Label 0 | Label 1 | Task groups |
|---|---:|---:|---:|---:|
| Train | 341 | 264 | 77 | 14 |
| Validation | 73 | 52 | 21 | 4 |
| Test | 88 | 66 | 22 | 4 |

The split is generated with seed `4266` by:

```bash
./.venv/bin/python scripts/split_dataset_502.py
```

`train.jsonl`, `validation.jsonl`, and `test.jsonl` are training-ready files.
Each row contains only `candidate_id`, `label`, and `model_input`. Annotation
reasons, review flags, source type, construction metadata, and split metadata
are deliberately excluded from model input.

`split_manifest.csv` records auditing metadata outside the model input.
`task_group_assignments.csv` gives the group-level assignment, and
`split_statistics.json` records counts, the random seed, and the source-file
SHA-256 hash.

## Leakage controls

- Every `task_group_id` stays in one split.
- Declared nested or derivative task families stay in one split. The family
  definitions and their reasons are in
  `data/manifests/dataset_502_v1_task_families.json`.
- Repeated `model_input_fingerprint` and `near_duplicate_group_id` values cannot
  cross splits.
- Tools that occur in only one independent task component remain in training,
  so the classifier is not evaluated on a tool schema it could not observe.
- Both labels occur in all three splits, and all eight tools occur in training.

## Scope limitation

All Label 1 rows in this version are constructed candidates, while all observed
rows are Label 0. The split prevents task-family leakage, but it does not remove
this source/label confound. Report results as a Workspace proof of concept, and
add observed positives plus constructed authorized negatives before making a
strong real-attack generalization claim.
