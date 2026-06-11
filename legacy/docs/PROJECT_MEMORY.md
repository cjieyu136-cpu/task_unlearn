# Project Memory

Last updated: 2026-05-15

This file is the central working memory for this project. It is meant to keep
the codebase, paper writing, and experiment interpretation aligned.

It does **not** treat every script in the repository as part of the final
method. Many files are exploratory, diagnostic, archival, or intermediate.

## 1. Project Identity

Base repository:

- Original paper/code: `Task-Aware Machine Unlearning with Application to Load Forecasting`
- Original repo mainline: centralized TAMU/PUMU for load forecasting
- Original entrypoints: `eval_unlearn.py`, `eval_unchange.py`, `gen_index.py`,
  `train_nn.py`

Current project direction:

- Extend the original repository toward federated load-forecasting repair after
  unlearning
- Focus on **federated neural repair realization**, not on rewriting the whole
  original TAMU paper
- Treat the codebase as a research workspace with multiple explored branches,
  then select a smaller final narrative for the paper

## 2. Important Principle

Not every existing code path should be preserved in the final paper.

This repository contains:

- formalized mainline code
- exploratory implementations
- audits
- ablations
- temporary comparison runners
- archived baselines

The paper should be faithful to the **final retained method logic**, not to the
entire exploration history.

## 3. Original Repo Capability

The original repository already supported:

- `linear`, `conv`, `mlpmixer`
- centralized unlearning / repair
- criteria such as `mse`, `mape`, `cost`
- affine-head style repair on top of neural backbones

This matters because some things that look like "our additions" are actually
extensions or reorganizations of existing repo ideas, not brand-new concepts.

## 4. Major Code Evolution We Introduced

These are the major directions added on top of the original repo during the
project:

1. Topology-aware layer
   - graph/topology utilities
   - topology-aware affine head
   - topology-aware Newton-style unlearning tools

2. Federated realization
   - federated versions of unlearning/repair runners
   - bus/client partition logic
   - block/local federated influence approximations

3. Runtime client/server abstraction
   - explicit client-side feature / gradient / score handling
   - server-side aggregation and repair orchestration
   - multiple feature/runtime modes for neural models

4. Enhancement modules
   - topology-aware partition and regularization
   - server-side fusion feature interface variants
   - shadow-weight and safety/joint-score style cost-oriented enhancements
   - secure-aggregation-style mock interface

## 5. What Is Exploration vs What Is Final-Mainline

### 5.1 Exploration / diagnostics / auxiliary work

These exist in the repository but should not automatically become paper method
components:

- many `run_fed_*_audit.py` scripts
- `eval_fed_cg_probe.py`
- communication summaries and CG scan tools
- archived prechange folders under `archive/`
- alternative branches kept for alignment or design comparison
- intermediate score shaping variants that were explored but not necessarily
  retained as the final story

These files are valuable evidence and debugging tools, but they are not all
equal to "the method".

### 5.2 Current retained paper mainline

Based on the current working consensus, the main paper should emphasize:

- a **federated neural repair framework**
- cloud-edge decoupled criterion-aware repair logic
- frozen backbone + server-side affine repair head
- two neural backbones as the main validated objects:
  - convolutional model
  - MLP-Mixer
- a unified repair pipeline:
  - criterion signal on test set
  - client-local response
  - global score aggregation
  - constrained reweight repair

Enhancement modules that may be retained in the final story:

- topology-aware partition / topology regularization
- server-side fusion feature interface
- cost-oriented shadow weighting

Topology should not be treated as a single all-or-nothing block. The current
retained interpretation is:

1. topology-aware client partition: retained and high-priority
2. topology-aware server-side fusion: retained and high-priority
3. topology-regularized affine repair head: retained and stable
4. topology-aware encoder propagation: exploratory / optional / later-stage
   enhancement, not the current default priority

## 6. Current Paper Boundary

The current paper should **not** be framed as "we preserve every branch that
ever existed in the repo".

Instead, the paper should be framed as:

- selecting and stabilizing one final method mainline from a larger research
  workspace
- presenting only the retained logic as the formal contribution

Recommended boundary statements:

- the paper focuses on the repair stage after complete unlearning
- the paper focuses on federated neural realization
- linear and other branches may remain as baseline, sanity check, or background,
  but are not necessarily the paper's main subject

## 7. Current Position on the Linear Path

Important nuance from recent discussion:

- the repository still contains a federated `linear` path
- however, the current paper does **not** need to center the narrative around
  `linear -> NN` expansion
- the preferred narrative is now:
  - unified problem setting
  - neural federated repair as the main retained realization
  - linear can be mentioned as baseline/sanity-check/background if needed

Do not force the final paper to mimic the original paper's linear-first logic
if that no longer matches the retained method identity.

## 8. Current Position on Runtime / Feature Modes

The codebase contains multiple neural feature/runtime modes, including:

- `precomputed_local_cache`
- `local_frozen_backbone`
- `topology_local_fusion`
- `local_mask_fusion`
- `fusion_topology_only`

This does **not** mean the paper must present all of them as equal-status
formal methods.

For the paper:

- choose the realization(s) that are actually retained
- mention alternatives only when needed for ablation or implementation detail
- avoid implying that every explored runtime mode is part of the final formal
  method definition

Current practical interpretation:

- `topology_local_fusion` can include topology injection at encoder + fusion
- `fusion_topology_only` / `local_mask_fusion` is the cleaner retained route
  when we want:
  - topology-aware partition
  - topology-aware fusion
  - topology-regularized repair
  - but **no** topology propagation inside the local encoder

## 9. Current Position on Fusion Interface

The server-side fusion feature interface is important, but its exact form is an
implementation choice inside the retained framework.

Code exploration includes layouts such as:

- `concat_residual`
- `concat_local`
- `secure_agg_mean`

Paper-writing rule:

- define the need for a server-side fusion feature interface at the framework
  level
- only promote a specific fusion layout to formal-method status if it is truly
  retained as the final chosen implementation
- otherwise present it as an implementation option or ablation

## 10. Current Position on Secure Aggregation

The current code includes only a **mock** secure aggregation layer.

This is an interface-level simulation, not a cryptographic secure aggregation
protocol.

Paper-writing rule:

- do not claim full secure aggregation implementation
- acceptable wording:
  - secure-aggregation-style interface
  - aggregate-only simulation
  - mock secure-sum mode

## 11. Current Position on Shadow Weight / Joint Score

Shadow-weight and related safety/joint-score logic came from iterative
exploration and later refinement.

Current interpretation:

- shadow weight is best treated as a **task-oriented enhancement layer**
- its strongest retained value is in cost-oriented repair
- it should not automatically be described as a universal enhancement for all
  criteria unless results clearly support that

If the final paper keeps only part of these variants, that is acceptable.

## 12. Paper Writing Guidance

The final paper should be loyal to the retained method, not to every script.

Recommended writing stance:

1. Main method
   - federated neural repair framework
   - criterion-aware cloud-edge repair flow

2. Retained implementation object
   - frozen backbone
   - server-side affine repair head
   - selected neural backbones

3. Retained enhancement layers
   - topology
   - fusion interface
   - cost-oriented shadow weighting

4. Non-mainline content
   - exploratory branches only as ablation, sanity check, or engineering note

Avoid:

- writing every explored branch as if it were part of the final theorem-level
  method
- forcing the paper to cover all repository artifacts
- overstating mock/system exploration code as finalized core contribution

## 13. Practical Review Rule

When checking whether a paper section is "consistent with the code", use this
decision order:

1. Is it consistent with the retained final method logic?
2. Is it supported by the code paths we still want to claim?
3. If it conflicts with exploratory-only code, ignore the exploratory code.
4. If it depends on a branch we no longer want to keep, remove or demote it in
   the paper.

Do **not** use "the repository contains a script" as the default criterion for
"the paper must include this".

## 14. Useful Anchors in the Repo

Original repo baseline:

- `readme.md`
- `eval_unlearn.py`
- `eval_unchange.py`
- `train_nn.py`

Topology layer:

- `utils/topology.py`
- `utils/topo_affine.py`
- `utils/topo_unlearn.py`

Federated neural repair path:

- `eval_fed_unchange.py`
- `eval_fed_unlearn.py`
- `utils/fed_data_utils.py`
- `utils/fed_tamu_pipeline.py`
- `utils/fed_client_runtime.py`
- `utils/fed_server_runtime.py`

Enhancement / auxiliary pieces:

- `utils/fed_topology_nn.py`
- `utils/fed_secure_aggregation.py`
- `utils/fed_repair_utils.py`
- `utils/fed_vjp_utils.py`
- `utils/index_utils.py`

Archived baseline and prechange checkpoints:

- `archive/2026-05-04_topology_fed_baseline/`
- `archive/2026-05-08_joint_score_prechange/`
- `archive/2026-05-08_matched_repair_prechange/`

## 15. Current Communication Rule for Future Work

When discussing this project in future turns:

- default to the retained final mainline
- explicitly label exploration as exploration
- do not conflate archival or diagnostic scripts with the final paper method
- if uncertain whether something is retained, check this document first and then
  verify with the user before elevating it into the paper narrative

## 16. Conversation History Summary

This section records the **actual decisions and corrections made in the current
chat**, not just the static project state.

### 16.1 Early paper-structure conclusion

A key early conclusion in this chat was:

- do **not** force the current paper to imitate the original paper's
  `linear -> NN` expansion logic
- instead use:
  - unified problem setting
  - different realization layers / model-family-specific implementations
  - discussion of neural-side design choices and enhancements

The user's earlier concern was whether `linear` had to be kept as the first
mainline because the original TAMU paper did that. The conclusion here was:

- that is no longer the best structure for the current work
- the current code and paper direction are better explained as a unified
  federated repair problem with different realizations

### 16.2 Linear vs NN positioning

Another explicit conclusion from this chat:

- `linear` should not be forcibly rewritten into the same federated runtime
  story as the neural path
- its value is its clarity, analytic structure, and baseline role
- the neural federated path is where the richer runtime / representation /
  fusion / enhancement logic lives

Working phrasing established in the chat:

- `linear`: shared-statistics federated baseline
- `NN`: representation-level federated runtime

This distinction matters for writing and for future reviews.

### 16.3 What had already been accomplished before the repo backtrace

In the chat we summarized that we had already done the following conceptual
work before inspecting the repository again:

1. Re-judged the paper structure
2. Decided not to force `linear` and `NN` into one fake-unified federated form
3. Resolved the concern that `linear` must occupy the same narrative role as in
   the original paper
4. Chosen a more IEEE-like organization:
   - unified formulation
   - separate realizations
   - experiments
   - discussion

These are conversation-level decisions and should be preserved even if later
file-level evidence is incomplete.

### 16.4 Repo backtrace result from this chat

Later in the chat we inspected the repository and reconstructed the main code
evolution:

1. original centralized TAMU/PUMU repo
2. topology-aware affine/topology-unlearning layer
3. federated unlearning / repair runners
4. neural runtime abstraction
5. enhancement modules such as fusion variants, shadow weighting, and secure
   aggregation mock

This backtrace was used to answer:

- what changes had been made relative to the original paper code

Important note from this chat:

- this backtrace shows the **exploration history**
- it does **not** automatically define the final paper scope

### 16.5 First review of the user's PDF and why it was partially wrong

During the first review of the user's modified paper PDF, the initial judgment
was too conservative because it leaned too much on the full repository content.

What happened:

- the paper was compared against too many existing code branches
- exploratory and archival branches were over-weighted
- this led to a stricter-than-intended judgment about whether the paper was
  "consistent with the code"

The user correctly pointed out:

- many files were only exploration
- the final paper does not need to preserve every explored path

This was an important correction in the chat.

### 16.6 Corrected review standard established in this chat

After the user's correction, the review standard was revised to:

- the paper should be compared against the **retained final method logic**
- not against every script or intermediate branch in the repository

This became one of the most important memory items from the chat.

Corrected rule:

- papers do not need to be responsible for all code
- papers need to be responsible for the final method they claim

### 16.7 Outcome of the corrected paper judgment

After that correction, the conversation-level conclusion became:

- the user's current paper direction is broadly aligned with the retained
  project mainline
- the main remaining task is not to "cover all code"
- instead it is to:
  - separate final method vs explored branches
  - avoid writing chosen implementations as if they were the only possible
    mathematical form
  - keep the narrative tightly centered on the retained federated neural repair
    framework

### 16.8 Why this memory file was requested

The user explicitly requested a memory document because:

- the assistant appeared to be forgetting conversation-specific decisions
- future chat windows may not have the full context
- the project needs one central document that preserves:
  - what was actually decided in conversation
  - what was explored in code
  - what should count as final retained mainline

This file is intended to solve that.

## 17. What the Assistant Currently Remembers from This Chat

The following points are the core retained memory from the current chat and
should be considered high-confidence unless the user later overturns them.

### 17.1 High-confidence retained memory

1. The original paper structure should not be copied mechanically.
2. The current project should be written around a federated neural repair
   framework.
3. `linear` still exists in code but is no longer required to dominate the
   narrative.
4. Many repo files are exploratory and should not automatically be treated as
   part of the final paper method.
5. The paper should match the final retained logic, not the whole exploration
   history.
6. The current retained neural focus is on:
   - conv
   - MLP-Mixer
   - frozen backbone + server affine repair head
   - criterion-aware cloud-edge repair flow
7. Topology / fusion interface / shadow weight are enhancement layers, not
   necessarily equal-status standalone methods.
8. Secure aggregation in the current code is mock/interface-level, not full
   cryptographic secure aggregation.
9. The four-layer topology idea has been revised: the currently preferred
   retained order is partition + fusion + repair as the stable topology path,
   while encoder-side topology is lower priority and may remain exploratory.

### 17.2 Medium-confidence memory

These are remembered from the chat but may need user confirmation before being
treated as final paper policy:

- whether the final paper should mention the linear federated path explicitly as
  a baseline, or mostly omit it
- which exact fusion interface should be elevated from implementation option to
  final claimed method component
- how much of joint-score / safety-gate logic remains in the final retained
  paper story

## 18. Future Use of This File

In future conversations, this file should be used for three purposes:

1. onboarding another chat/session
   - what project this is
   - what code history exists
   - what final narrative is being retained

2. reviewing paper drafts
   - compare draft text against retained mainline
   - not against every exploratory file

3. detecting drift
   - if a later answer starts treating all code as final method content, that is
     a regression relative to this chat's agreed standard

## 19. Current Known Limitation of This Memory File

This file summarizes the chat and the code interpretation, but it is still a
human-curated memory artifact.

It does not replace:

- checking actual code when a precise implementation claim matters
- checking actual experiment outputs when making quantitative claims
- asking the user to confirm when a previously explored branch is being promoted
  into the final paper narrative
