# CG Scan Plan

Last updated: 2026-05-04

## Goal

Tune CG-based inverse HVP settings for the topology-aware federated path while preserving:

- good unlearning / repair quality
- small fed-vs-global repair deltas
- acceptable residuals
- lower runtime than the current conservative baseline

## Current Baseline

Current validated baseline settings:

```text
+feature_mode=topology_local_fusion
+gnh=true
+cg_tol=1e-7
+cg_maxiter=4000
+damping=1e-4
+block_damping=1e-4
```

Observed behavior:

- result quality is good
- CG often reaches `4000` iterations
- residuals are usable but not elegant
- runtime is high

## What to Measure

For each run, record:

1. quality
   - `metric_mse_test`
   - `metric_mape_test`
   - `metric_cost_test`
2. federated closeness
   - `delta_metric_mse_test_fed_minus_global`
   - `delta_metric_mape_test_fed_minus_global`
   - `delta_metric_cost_test_fed_minus_global`
   - `M_relative_l2_error`
   - `score_relative_l2_error`
3. CG diagnostics
   - `cg_iterations`
   - `cg_relative_residual`
   - `global_cg_iterations`
   - `global_cg_relative_residual`
   - `fed_cg_iterations`
   - `fed_cg_relative_residual`
4. runtime
   - wall-clock runtime per run

## Scan Order

We should scan in this order:

### Stage A: damping scan

Keep:

```text
+cg_tol=1e-7
+cg_maxiter=4000
+gnh=true
```

Scan:

```text
+damping in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
+block_damping = same as damping
```

Target:

- lower residual
- preserve fed-vs-global closeness

### Stage B: tolerance scan

Fix the best damping from Stage A, then scan:

```text
+cg_tol in [1e-5, 3e-6, 1e-6, 3e-7, 1e-7]
```

Target:

- see whether we can stop earlier without hurting closeness

### Stage C: maxiter scan

Fix damping and tolerance, then scan:

```text
+cg_maxiter in [1000, 1500, 2000, 3000, 4000]
```

Target:

- find the smallest practical cap

### Stage D: GNH ablation

Run:

```text
+gnh=true
+gnh=false
```

Target:

- test whether `gnh=true` is actually necessary for topology-local-fusion

## Minimal Experiment Matrix

To keep cost reasonable, use:

- models: `conv`, `mlpmixer`
- task: `eval_fed_unchange.py`
- criteria: first `mse`, then `cost`
- index mode: `helpful`

Recommended order:

1. `conv + mse`
2. `mlpmixer + mse`
3. `conv + cost`
4. `mlpmixer + cost`

Do not start with `mape`; it is useful, but lower priority for numerical scanning.

## Success Criteria

A candidate setting is considered good if:

1. `fed-vs-global` test deltas remain very small
2. `M_relative_l2_error` and `score_relative_l2_error` stay near baseline
3. `cg_relative_residual` improves or remains stable
4. runtime is meaningfully lower than the baseline

## Immediate Recommendation

Start with Stage A on:

- `conv + mse`
- `mlpmixer + mse`

Only after that, promote the best candidate to `cost`.
