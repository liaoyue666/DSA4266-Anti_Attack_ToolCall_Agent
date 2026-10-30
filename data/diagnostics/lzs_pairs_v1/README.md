# Pending development probe review

This directory contains 20 synthetic candidates, arranged as 10 pairs from
two template families with address variants. They are developer probes,
not ten independent AgentDojo tasks and not a final challenge benchmark.

- [candidates.jsonl](candidates.jsonl): every label is null. Only model_input
  may be supplied to a model; candidate IDs are tracking fields.
- [review_metadata.jsonl](review_metadata.jsonl): expected labels and reasons
  for review assistance only; all records remain pending with no reviewer.
- [review_sheet.csv](review_sheet.csv): complete context and proposed actions
  for human review; UTF-8 BOM for spreadsheet applications.

Each pair holds the request and observations fixed and changes only one
recipient/participant. Injection text repeats the authorized address but
refers to the unauthorized address by alias. Review each complete case before
adopting a label. Expected labels are construction hypotheses, not ground truth.

Do not append these probes to the 502-row dataset or use them for real-episode
ASR. Any future formal dataset incorporation requires adjudication, provenance
tracking and keeping each pair and its derivative family in one split.

Regenerate a separate draft using scripts/build_diagnostic_pairs.py. The
generated review sheet is a starting point, not a master annotation source for
the existing 502-row delivery.
