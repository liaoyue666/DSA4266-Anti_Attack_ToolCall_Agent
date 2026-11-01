# Public Workspace 509 v1: grouped cross-validation

## Dataset and protocol

- 509 confirmed reviewed calls: 400 authorized and 109 unauthorized.
- Sources: 120 rows from `public_workspace_120_v1` and 389 confirmed rows from
  `public_workspace_400_v1`; 11 unresolved rows are excluded.
- Four-fold leave-task-families-out cross-validation over eight task families.
- Each row receives one out-of-fold prediction. Rows from the same task family never
  appear in both training and validation for a fold.
- These results are development results, not an independent held-out benchmark.

## Results at the fixed 0.5 threshold

| Model | PR-AUC | Unauthorized recall | Unauthorized F1 | False-block rate |
|---|---:|---:|---:|---:|
| Majority (always authorized) | 0.214 | 0.000 | 0.000 | 0.000 |
| TF-IDF + logistic regression | 0.536 | 0.780 | 0.338 | 0.773 |
| Rule/policy filter | 0.750 | 0.761 | 0.557 | 0.265 |
| Frozen DistilRoBERTa + dense head (3-seed ensemble) | 0.451 | 0.936 | 0.399 | 0.750 |

The frozen Transformer catches most unauthorized calls but blocks too many authorized
calls. Its confusion matrix is TN=100, FP=300, FN=7, TP=102. The rule baseline is the
strongest current method.

## Important limitation

The folds are intentionally strict and materially distribution-shifted. In fold 1,
the held-out task families contain all 207 `append_to_file` calls while the fold's
training data contains none. Fold 1 also contributes 307 of the 509 out-of-fold rows.
Consequently, the micro result is dominated by a large unseen-tool/task-family slice.
This is useful as a robustness stress test, but the project should report the per-fold
results and add broader task/tool coverage before treating it as an operational estimate.
