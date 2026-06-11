# Federated Unlearning Optimization Roadmap

Last updated: 2026-05-04

## Baseline Archive

Current validated baseline has been archived at:

- `D:\ProgramFiles\Project_4\task_aware_machine_unlearning-master\archive\2026-05-04_topology_fed_baseline`

Archive contents:

- `code/`: key source files for the validated topology-aware federated baseline
- `results_flat/`: flattened copies of the main summary CSVs
- `results_manifest.csv`: mapping from flattened result filenames to original paths

This baseline corresponds to:

- `linear` federated path working
- `conv` / `mlpmixer` topology-aware `fed_unlearn` working
- `conv` / `mlpmixer` topology-aware `fed_unchange` working for `mse`, `mape`, `cost`
- retrain baseline in `eval_fed_unlearn.py` fixed to use the same topology-affine training definition

## Project Goal

Improve the current topology-aware federated unlearning prototype toward a more industrial federated system while preserving:

- `linear`, `CNN`, and `MLPMixer`
- topology-aware client partition / encoder / fusion / repair
- repair quality close to centralized
- CG-based inverse HVP as the mainline method

## Guiding Principles

1. Do not break the validated baseline.
2. Separate algorithm improvements from system improvements.
3. Prefer measurable progress: every phase should end with artifacts and summaries.
4. Keep centralized-vs-federated comparability throughout.

## Phase 0: Lock Baseline and Build Comparison Harness

### Goal
Make future modifications easy to compare against the archived baseline.

### Tasks

1. Create a single summary script that aggregates:
   - `linear`
   - `conv`
   - `mlpmixer`
   - `fed_unlearn`
   - `fed_unchange`
   - `mse`, `mape`, `cost`
2. Normalize output columns across result files.
3. Add a "baseline vs current" comparison table generator.

### Success Criteria

- One command can regenerate a concise experiment summary table.
- Future modifications can be judged against baseline numerically.

## Phase 1: Industrialize Execution Without Changing the Core Method

### Goal
Upgrade the current single-process research workflow into a cleaner federated execution workflow.

### Tasks

1. Introduce explicit client/server execution objects:
   - client worker
   - server coordinator
   - message schema
2. Separate debug mode and strict federated mode:
   - debug mode may save per-client payloads
   - strict mode should save only aggregated quantities unless explicitly enabled
3. Add incremental checkpointing:
   - cache train/test fusion features
   - cache unlearn indices
   - cache global gradients / scores / IHVP outputs
   - save one row per `l1_constraint` immediately after it completes
4. Add resumable execution:
   - detect existing partial outputs
   - skip completed stages

### Success Criteria

- Long runs can be resumed.
- Repeated runs avoid recomputing frozen features.
- Result directories expose clear stage completion.

## Phase 2: Make the Federated Boundary More Realistic

### Goal
Reduce the gap between the current simulation and a stricter industrial federated setting.

### Tasks

1. Replace secure-aggregation mock with a stricter aggregate-only interface:
   - clients emit masked or aggregate-safe summaries
   - server does not keep per-client raw repair payloads in strict mode
2. Track communication volume:
   - bytes per client upload
   - bytes per server broadcast
   - total round communication
3. Add client availability simulation:
   - random dropout
   - missing client handling
   - fallback aggregation policy
4. Add non-IID data options:
   - temporal skew
   - regional imbalance
   - client sample count imbalance

### Success Criteria

- Strict mode hides per-client internals from normal outputs.
- Communication and availability metrics appear in result summaries.
- The system can run under client dropout and non-IID settings.

## Phase 3: Improve Numerical Efficiency While Preserving CG

### Goal
Keep CG as the formal method path, but reduce runtime and improve numerical behavior.

### Tasks

1. Add CG diagnostics summary tables:
   - iteration count
   - residual norm
   - relative residual
   - per-criteria distribution
2. Add parameter sweeps for:
   - `damping`
   - `block_damping`
   - `cg_tol`
   - `cg_maxiter`
   - `gnh`
3. Add optional preconditioning hooks for CG.
4. Investigate block-structured or topology-aware preconditioners.
5. Keep exact affine oracle path for debug only.

### Success Criteria

- We know which settings are fastest while preserving closeness.
- CG still matches centralized repair closely with lower runtime.

## Phase 4: Strengthen the Topology Story

### Goal
Turn topology usage from "correct design choice" into a clear paper contribution.

### Tasks

1. Add topology ablations:
   - no topology
   - topology partition only
   - topology partition + fusion
   - full topology-aware path
2. Add client partition ablations:
   - contiguous partition
   - balanced topology partition
   - random partition
3. Add client-count sensitivity:
   - `2 / 4 / 7 / 14` clients
4. Add optional topology-aware reweighting:
   - region-sensitive repair emphasis
   - boundary-bus emphasis

### Success Criteria

- Experimental tables show where topology helps.
- Partition and fusion choices become defendable contributions.

## Phase 5: Paper-Ready Experimental Package

### Goal
Produce a final experiment set suitable for thesis/report/paper use.

### Required Tables

1. Main performance table:
   - centralized vs federated
   - `linear`, `conv`, `mlpmixer`
   - `mse`, `mape`, `cost`
2. Unlearning completeness table:
   - original
   - direct unlearn
   - retrain
   - prediction distance to retrain
3. Repair closeness table:
   - global repair
   - fed repair
   - parameter and metric deltas
4. Ablation table:
   - topology components
   - client partition choices
5. System table:
   - runtime
   - communication
   - cache savings
   - dropout robustness

### Success Criteria

- All main claims are backed by one dedicated table.
- Report text can be written directly from the generated summaries.

## Recommended Execution Order

We should proceed in this order:

1. Phase 0: summary harness
2. Phase 1: caching + resumable execution
3. Phase 3: CG efficiency tuning
4. Phase 4: topology ablations
5. Phase 2: stricter federated boundary and system realism
6. Phase 5: paper-ready package

This order keeps the validated baseline intact while improving speed and evidence first.

## Immediate Next Step

Start with **Phase 0 + the first half of Phase 1**:

1. build a unified result summarizer
2. add feature/result caching
3. add incremental `l1_constraint` result writes

These changes offer the highest leverage with the lowest risk.
