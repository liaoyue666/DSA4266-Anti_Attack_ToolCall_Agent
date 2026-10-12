# Learning to Audit Tool Calls Against Prompt Injection

This repository contains the DSA4266 project on protecting tool-using LLM agents from **indirect prompt injection**. The project studies whether a trained, context-aware verifier can detect unauthorized tool calls before execution while preserving the agent's ability to complete legitimate user tasks.

> **Project status:** proposal and implementation planning. Experimental results and exact reproduction commands will be added as the system is implemented.

## Motivation

Tool-using agents may read attacker-controlled content from emails, documents, or other external sources. If the agent mistakes this untrusted content for a user instruction, it may issue a consequential tool call that the user never authorized.

The project asks:

> Can a trained, context-aware verifier block unauthorized tool calls while preserving legitimate task completion?

The main contribution will be a trained authorization model, its integration at the tool-execution boundary in AgentDojo, and a controlled evaluation of the resulting security–utility trade-off.

## Threat model

We consider an agent that receives a legitimate user request, observes potentially attacker-controlled text, and proposes a tool call. An attack succeeds when the untrusted content causes the agent to perform an action that is not authorized by the user's original request.

The verifier does not attempt to sanitize all natural-language input. Instead, it evaluates the proposed action immediately before a consequential tool is executed.

## Proposed system

```mermaid
flowchart LR
    U[Original user request] --> A[Tool-using agent]
    O[Tool observation<br/>marked as untrusted] --> A
    A --> P[Proposed tool call<br/>name and arguments]
    U --> V[Authorization verifier]
    O --> V
    P --> V
    V -->|ALLOW| T[Execute tool]
    V -->|BLOCK| R[Standardized refusal<br/>agent may continue]
```

For every proposed action, the verifier receives:

- the original user request;
- the proposed tool name and arguments; and
- a short, relevant tool observation explicitly marked as untrusted.

It outputs the probability that the action is unauthorized:

| Label | Meaning |
|---:|---|
| `0` | Authorized by the user's request |
| `1` | Unauthorized action |

A validation-set threshold converts this score into an `ALLOW` or `BLOCK` decision.

## Model design

The primary verifier uses a frozen pretrained Transformer encoder to represent the relationship between user intent, untrusted text, and the proposed action. A small trainable classification head performs binary authorization:

```text
combined sequence
    → frozen Transformer encoder
    → Dense(64, ReLU)
    → Dropout
    → Dense(1, sigmoid)
```

Training uses mini-batches, Adam, and binary cross-entropy. Dropout or L2 regularization will be used to reduce overfitting. Class weighting and per-class metrics will be considered if the authorization labels are imbalanced.

If the pilot dataset is large enough, a learned-embedding + BiGRU classifier will be included as a sequence-model baseline.

## Data construction

Candidate calls will be collected from benign and attacked AgentDojo development trajectories. Additional attacker-goal candidates may be constructed when needed.

To improve label quality and prevent leakage:

- two team members will verify a stratified sample of labels;
- ambiguous examples will be recorded separately;
- dataset statistics will report class balance, action types, exclusions, and final counts;
- AgentDojo's episode-level attack outcome will **not** automatically become the label of every call; and
- train, validation, and test splits will be separated by complete user task and attack scenario where feasible, preventing near-duplicate requests or injection templates from crossing splits.

All model and threshold choices will use only the training and validation sets. Held-out episodes will be reserved for final evaluation.

## Experimental conditions

| Condition | Purpose |
|---|---|
| No defense | Establish baseline task success and prompt-injection attack success |
| Rule/policy filter | Measure protection achievable without a learned model |
| Trained verifier | Evaluate the proposed context-aware authorization classifier |
| Prompt-only LLM auditor | Compare against an untrained LLM decision using the same context |

The controlled experiment will keep the AgentDojo task suite, agent model, eligible task/attack pairs, tool set, prompt, and step limit fixed across conditions. Every verifier decision will be logged. A blocked call will produce a standardized refusal so that the agent can continue the task when possible.

### Ablation studies

Two context ablations will test what information the verifier actually relies on:

1. remove the original user request;
2. remove the untrusted tool observation.

## Evaluation

Evaluation is performed at two levels.

### Episode-level metrics

- **Attack Success Rate (ASR):** lower is better.
- **Benign user-task success:** higher is better.

### Call-level metrics

- precision, recall, and F1 for unauthorized calls;
- false-block rate on authorized calls; and
- added latency and cost.

The decision threshold will be selected on validation data to balance unauthorized-call recall against false blocks. Planned analyses include:

- training and validation loss curves;
- a held-out confusion matrix;
- ASR versus benign task success;
- latency and cost overhead;
- missed-attack analysis; and
- analysis of legitimate calls blocked because of superficial cues.

A model that blocks every action is not considered secure or useful; the central objective is to improve security without destroying task utility.

## Reproducibility plan

The final repository will record:

- dataset construction and exclusion rules;
- split-generation logic and random seeds;
- model and optimizer configurations;
- the validation procedure used to choose the blocking threshold;
- the AgentDojo task suite, agent configuration, prompts, tool set, and step limit;
- per-call verifier decisions; and
- scripts for reproducing tables and figures.

## Scope and limitations

The core deliverable is one trained verifier, two comparator defenses, and a reproducible AgentDojo evaluation. The Transformer backbone remains frozen in the primary experiment to limit data and compute requirements.

The project does not attempt to train a foundation model from scratch. If verified labels are too scarce, the model will be simplified and the limitation reported rather than making unsupported claims. Conclusions will be restricted to the evaluated tasks, tools, agent model, and attack conditions.

## Responsible use

All attacks and tool executions should be evaluated in a controlled, sandboxed environment. This repository is intended for defensive research into authorization and prompt-injection robustness, not for deployment against systems without permission.

## Planned repository structure

```text
.
├── configs/          # Experiment and model configurations
├── data/             # Dataset metadata and split definitions
├── src/              # Verifier, baselines, and AgentDojo integration
├── scripts/          # Training and evaluation entry points
├── tests/            # Unit and integration tests
├── results/          # Generated metrics and figures
└── README.md
```

The structure above is a plan and will be updated as implementation files are added.

## Milestones

- [ ] Verify the AgentDojo interception hook and decision logging.
- [ ] Collect and audit call-level authorization labels.
- [ ] Establish no-defense and rule-based baselines.
- [ ] Train and validate the primary verifier.
- [ ] Evaluate the prompt-only LLM auditor.
- [ ] Run context ablations and threshold analysis.
- [ ] Complete held-out evaluation and failure analysis.
- [ ] Publish reproducible configurations, figures, and results.

## Team

DSA4266 group project. Team-member names and responsibilities will be added here.

