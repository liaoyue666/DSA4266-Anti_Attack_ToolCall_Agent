# Dataset delivery: dataset-502-v1

This directory contains 502 unique annotation candidates: 382 observed calls
and 120 constructed candidates. These are candidates for human review, not
already annotated or adjudicated data. Every `label` is `null`.

The 120 constructed rows carry an `expected_unauthorized` review intent. That
intent is a hypothesis for review, **not a confirmed Label 1**. Constructed
rows were not proposed or executed by an agent and must not be included in
real-episode ASR calculations. Their construction explanations are isolated in
`review_metadata.jsonl` and are not part of `model_input`.

`observed_calls.jsonl` is deduplicated by the canonical model-input fingerprint
(original request, ordered pre-call observations, tool name, and arguments).
Each row retains all source call/episode references for exact duplicate source
rows. `constructed_calls.jsonl` is copied from the existing delivery-001 file;
it is not regenerated. Only review records whose IDs match those 120 delivered
constructed candidate IDs are included. The other 76 records in the source
196-row review list are excluded.

For every candidate, model input consists only of `original_user_request`,
`observations_before_call`, and `proposed_action` (`tool_name`, `arguments`).
Source IDs, task grouping, labels, duplicate groups, and all review metadata
remain outside that object.

Near-duplicate groups are retained and reported; 502 rows must not be
described as 502 independent tasks. When making any train/validation/test
split, keep every candidate with the same `task_group_id` together. For a
deduplicated row listing multiple `related_task_group_ids`, treat those groups
as linked and assign them to the same split as well; all constructed children
must remain with their parent task group. The final evaluation split still
requires team confirmation.

No API calls, collection runs, tool executions, or automatic labeling were
performed to prepare this package.
