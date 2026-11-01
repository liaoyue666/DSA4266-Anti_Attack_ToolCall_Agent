# Frozen DistilRoBERTa improvement experiments

## Protocol

All results use the same four outer task-family folds over Public Workspace 509 v1.
The encoder remains frozen. Three dense-head seeds are probability-ensembled. Fixed
threshold variants use 0.5. The final variant selects a separate threshold for each
outer fold using leave-one-task-family-out predictions from that fold's training data;
the outer validation family is never used for threshold selection.

The multi-field representation mean-pools the request, selected observations, and
proposed action separately, then concatenates those embeddings with absolute
differences and element-wise products. The rule-enhanced representation appends four
boolean policy features.

## Results

| Configuration | PR-AUC | Recall | F1 | False-block rate |
|---|---:|---:|---:|---:|
| Previous CLS, full class weight | 0.451 | 0.936 | 0.399 | 0.750 |
| Mean pooling, full class weight | 0.436 | 0.908 | 0.406 | 0.700 |
| Mean pooling, square-root class weight | 0.388 | 0.789 | 0.433 | 0.505 |
| Mean pooling, no class weight | 0.386 | 0.321 | 0.365 | 0.120 |
| Multi-field, no class weight | 0.454 | 0.771 | 0.553 | 0.278 |
| Multi-field + rule features, no class weight | **0.580** | 0.807 | **0.622** | **0.215** |
| Multi-field + rules, inner-selected thresholds | **0.580** | **0.835** | 0.621 | 0.233 |

## Interpretation

Mean pooling alone does not improve ranking. Reducing the positive-class weight moves
the operating point toward fewer false blocks, but also reduces recall and does not
improve PR-AUC. Thus class weighting explains part of the original over-blocking but
is not the main representation problem.

Separate field encoding materially improves F1 and false-block rate. Adding the four
transparent rule features produces the strongest learned verifier, improving PR-AUC
from 0.451 to 0.580 and reducing false blocks from 75.0% to 21.5%. Leakage-free inner
threshold selection reaches the target recall above 80%, but increases false blocks
slightly and cannot change PR-AUC.

The improved frozen verifier does not beat the fixed rule policy (PR-AUC 0.750) or its
high-precision ablation (F1 0.758, false-block rate 1.0%). Fold 1 remains the main
bottleneck: it contains the large held-out weekly-todos family and all 207
`append_to_file` calls, yielding PR-AUC 0.501 even for the best learned representation.
More task/tool coverage is needed before further head tuning is likely to produce a
robust state-of-the-art result.
