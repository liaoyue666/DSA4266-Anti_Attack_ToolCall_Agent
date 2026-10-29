# Preliminary offline verifier results

These are proof-of-concept results on `dataset-502-v1`, not evidence of
generalization to real successful prompt-injection attacks. All 120 Label 1
examples are constructed and all 382 observed calls are Label 0.

## Protocol

- The leakage-aware task-family split is used without modification.
- Model and threshold choices use training and validation data only.
- The held-out test split is evaluated after threshold selection.
- The positive class is unauthorized (`Label 1`).
- The threshold policy first requires validation unauthorized recall of at
  least 0.80, then minimizes false-block rate and breaks ties by F1.
- Transformer results use three seeds (`4266`, `4267`, and `4268`).

## Held-out test results

| Model | Precision (1) | Recall (1) | F1 (1) | False-block rate | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Majority Label 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.250 |
| TF-IDF + logistic regression | 0.339 | 0.864 | 0.487 | 0.561 | 0.594 |
| Frozen DistilRoBERTa + dense head | 0.245 | 0.727 | 0.367 | 0.747 | 0.516 |
| Fixed lexical policy | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |

The Transformer row reports the mean over three seeds with CLS pooling. The
dense head is `768 -> 64 -> 1`, with ReLU and dropout. DistilRoBERTa remains
frozen. Mean pooling was also tested and gave a lower mean test PR-AUC of about
0.287, versus about 0.516 for CLS pooling.

## Why the perfect rule result is not a generalization result

The lexical policy uses four transparent features fixed before test
evaluation: presence of injection cues, mismatch between requested and proposed
tool intent, action email absent from the request, and exact repetition of an
action target inside an injection block.

On the test split:

- all 22 constructed Label 1 rows repeat an action target inside the injection;
- none of the 66 observed Label 0 rows do;
- 59 of 66 observed Label 0 rows still contain an injection;
- 34 of 66 observed Label 0 rows use an email not explicitly written in the
  user request.

Therefore the rule does more than detect injection presence, but the perfect
separation directly matches how the unauthorized counterfactuals were
constructed. It may be a useful narrow policy guard, but it has not been tested
against observed successful attacks, paraphrased attack goals, indirect target
references, or constructed authorized controls.

## Error analysis

The frozen Transformer has substantial task-level shift. For seed 4266 it
false-blocks 95.2% of authorized `send_email` calls, while missing all six
unauthorized calendar calls in `workspace:user_task_4`. TF-IDF is less extreme
but still false-blocks 85.7% of authorized `send_email` calls. This indicates
that both learned models rely heavily on task and surface-form correlations.

Detailed local artifacts are generated under `results/error_analysis/` by:

```bash
./.venv/bin/python scripts/analyze_model_errors.py \
  --predictions results/frozen_distilroberta_cls/seed_4266/test_predictions.jsonl \
  --output-dir results/error_analysis/distilroberta_cls
```

## Decision before AgentDojo deployment

Do not deploy the current learned verifier as the primary AgentDojo gate: its
held-out false-block rate would severely reduce benign task completion. The
fixed lexical policy can be retained as a comparator, but its perfect offline
score must be reported together with the construction confound.

The highest-value next data additions are:

1. observed unauthorized calls from genuinely successful or partially
   successful attacked episodes;
2. constructed authorized controls produced with the same editing pipeline as
   the constructed unauthorized calls;
3. unauthorized examples that paraphrase or indirectly refer to the injection
   target rather than copying it exactly;
4. cross-review of each added example, followed by a regenerated task-family
   split.

After those additions, rerun all comparators unchanged before considering a
more complex encoder or partial Transformer fine-tuning.
