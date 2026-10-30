# LZS train/validation diagnostics

These committed artifacts come from the local diagnostic run described in
[the analysis](../../lzs_development_diagnostics.md). No test predictions,
model binaries, credentials or paid auditor results are included.

- [Summary](summary.json): four TF-IDF ablations and provisional probe comparisons.
- [Per-ablation metrics and validation predictions](tfidf/): reproducible
  validation threshold choices, versions and input hashes.
- [Validation errors by task/tool](validation_breakdown.json).
- [Validation threshold curve](validation_threshold_curve.json).
- [Action-only feature weights](action_only_top_features.json).
- [Tokenizer audit summary](token_audit_summary.json).

All original labels are provisional. Validation is also used to select
thresholds. The synthetic probe comparisons use pending expected labels,
not adjudicated gold labels; they are not real attack-success measurements.
The scripts regenerate full runtime artifacts under git-ignored results/.
