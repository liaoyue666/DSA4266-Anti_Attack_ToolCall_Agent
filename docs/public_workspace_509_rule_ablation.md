# Rule-policy ablation on Public Workspace 509 v1

All variants use the same 509 reviewed calls and the fixed threshold 0.5. The
leave-one-out variants retain the original weights rather than renormalizing them.

## Leave-one-out results

| Variant | PR-AUC | Recall | F1 | False-block rate |
|---|---:|---:|---:|---:|
| Full policy | 0.750 | 0.761 | 0.557 | 0.265 |
| Without injection cue | 0.737 | 0.367 | 0.537 | 0.000 |
| Without tool-intent mismatch | 0.723 | 0.633 | 0.758 | 0.010 |
| Without email-not-in-request | 0.602 | 0.761 | 0.557 | 0.265 |
| Without target-repeated-in-injection | 0.572 | 0.495 | 0.408 | 0.255 |

## Interpretation

`action_target_repeated_in_injection` is the strongest individual ranking signal:
it appears in 69 of 109 unauthorized calls and only 4 of 400 authorized calls. Its
individual PR-AUC is 0.677. Removing it produces the largest PR-AUC decrease and a
substantial recall decrease.

`tool_intent_mismatch` is responsible for most false blocks. Removing it reduces the
false-block rate from 26.5% to 1.0% and raises F1 from 0.557 to 0.758, at the cost of
reducing recall from 76.1% to 63.3%. This reduced policy is a useful high-precision
baseline but misses more unauthorized calls.

The email feature changes ranking (and therefore PR-AUC) but does not change any
0.5-threshold decisions in the full-policy ablation. The injection cue is common in
both classes (263 authorized and 101 unauthorized rows), so it is not sufficiently
discriminative by itself; its main role is to combine with other features to cross the
decision threshold.

Individual-feature variants cannot reach the 0.5 threshold because every individual
weight is below 0.5. Their precision/recall/F1 at that threshold are therefore zero;
their PR-AUC values, which measure score ranking, remain informative.
