# Repository Cleanup Plan

Last updated: 2026-05-16

## Goal

Shrink the repository to one paper-mainline:

- federated neural repair after complete unlearning
- frozen backbone + server-side affine repair head
- optional retained enhancements:
  - topology-aware partition / fusion / regularization
  - event-based unlearning index selection
  - cost-oriented weighting

Everything else should be either:

- moved to `legacy/`
- moved to `artifacts/`
- deleted from the repo working tree

## Target Layout

The cleaned repository should use this structure:

```text
task_aware_machine_unlearning-master/
├── conf/
├── core/
│   ├── clean_data.py
│   ├── train_nn.py
│   ├── gen_index.py
│   ├── gen_event_index.py
│   ├── eval_fed_unlearn.py
│   ├── eval_fed_unchange.py
│   └── func_operation.py
├── utils/
├── legacy/
│   ├── entrypoints/
│   ├── runners/
│   ├── summaries/
│   ├── tables/
│   ├── modules/
│   ├── tests/
│   └── docs/
├── artifacts/
│   ├── archive/
│   ├── deliverables/
│   ├── outputs/
│   ├── simulation_result/
│   ├── tmp/
│   └── papers/
├── data/
├── influence/
├── trained_model/
├── torch_influence/
├── readme.md
├── .gitignore
└── REPO_CLEANUP_PLAN.md
```

## Directory Creation

Create these directories first:

```powershell
New-Item -ItemType Directory -Force core
New-Item -ItemType Directory -Force legacy
New-Item -ItemType Directory -Force legacy\entrypoints
New-Item -ItemType Directory -Force legacy\runners
New-Item -ItemType Directory -Force legacy\summaries
New-Item -ItemType Directory -Force legacy\tables
New-Item -ItemType Directory -Force legacy\modules
New-Item -ItemType Directory -Force legacy\tests
New-Item -ItemType Directory -Force legacy\docs
New-Item -ItemType Directory -Force artifacts
New-Item -ItemType Directory -Force artifacts\archive
New-Item -ItemType Directory -Force artifacts\deliverables
New-Item -ItemType Directory -Force artifacts\outputs
New-Item -ItemType Directory -Force artifacts\simulation_result
New-Item -ItemType Directory -Force artifacts\tmp
New-Item -ItemType Directory -Force artifacts\papers
```

## Move Plan

### 1. Move mainline entrypoints into `core/`

```powershell
Move-Item clean_data.py core\
Move-Item train_nn.py core\
Move-Item gen_index.py core\
Move-Item gen_event_index.py core\
Move-Item eval_fed_unlearn.py core\
Move-Item eval_fed_unchange.py core\
Move-Item func_operation.py core\
```

After this move, update imports and any script paths that still assume root-level locations.

Files likely needing path/import updates:

- `core/eval_fed_unlearn.py`
- `core/eval_fed_unchange.py`
- `core/gen_index.py`
- `core/gen_event_index.py`
- `core/train_nn.py`
- `core/clean_data.py`
- `utils/reweight_utils.py`
- `readme.md`
- any `.ps1` or `.cmd` script you choose to keep

### 2. Move old centralized / intermediate entrypoints into `legacy/entrypoints/`

```powershell
Move-Item eval_fed_linear_unchange.py legacy\entrypoints\
Move-Item eval_fed_cg_probe.py legacy\entrypoints\
Move-Item eval_retrain_distance.py legacy\entrypoints\
Move-Item eval_topo_unchange.py legacy\entrypoints\
Move-Item eval_topo_unchange_repair_sign.py legacy\entrypoints\
Move-Item eval_topo_unchange_repo_style.py legacy\entrypoints\
Move-Item eval_topo_unchange_repo_style_fixedcrit.py legacy\entrypoints\
```

### 3. Move experiment runners into `legacy/runners/`

```powershell
Move-Item run_fed_block_influence_audit.py legacy\runners\
Move-Item run_fed_block_tamu_repair.py legacy\runners\
Move-Item run_fed_bus_tamu_audit.py legacy\runners\
Move-Item run_fed_bus_tamu_repair.py legacy\runners\
Move-Item run_fed_linear_block_audit.py legacy\runners\
Move-Item run_fed_local_feature_audit.py legacy\runners\
Move-Item run_fed_local_head_audit.py legacy\runners\
Move-Item run_fed_local_head_repair.py legacy\runners\
Move-Item run_fed_partial_input_audit.py legacy\runners\
Move-Item run_fed_runtime_block_audit.py legacy\runners\
Move-Item run_fed_tamu_repair.py legacy\runners\
Move-Item run_fed_vjp_audit.py legacy\runners\
Move-Item run_fed_vjp_repair_test.py legacy\runners\
Move-Item run_fixedcrit_repo_style.py legacy\runners\
Move-Item run_retrain_distance_repo_style.py legacy\runners\
```

Optional shell wrappers to move as well:

```powershell
Move-Item run_cg_probe_scan.ps1 legacy\runners\
Move-Item run_conv_topology_ablation.ps1 legacy\runners\
Move-Item run_full_mse_followup.ps1 legacy\runners\
Move-Item run_mlpmixer_alpha_scan.cmd legacy\runners\
Move-Item run_mlpmixer_alpha_scan.ps1 legacy\runners\
Move-Item run_model_compatibility_audit.ps1 legacy\runners\
Move-Item run_topology_ablation_pipeline.ps1 legacy\runners\
```

### 4. Move summary scripts into `legacy/summaries/`

```powershell
Move-Item summarize_cg_diagnostics.py legacy\summaries\
Move-Item summarize_cg_scan_results.py legacy\summaries\
Move-Item summarize_fed_block_tamu_repair.py legacy\summaries\
Move-Item summarize_fed_bus_tamu_repair.py legacy\summaries\
Move-Item summarize_fed_communication.py legacy\summaries\
Move-Item summarize_fed_local_head_repair.py legacy\summaries\
Move-Item summarize_fed_pipeline_results.py legacy\summaries\
Move-Item summarize_fed_tamu_repair.py legacy\summaries\
Move-Item summarize_fed_vjp_audit.py legacy\summaries\
Move-Item summarize_fed_vjp_repair_test.py legacy\summaries\
Move-Item summarize_fixedcrit_repo_style.py legacy\summaries\
Move-Item summarize_repair_sign.py legacy\summaries\
Move-Item summarize_repo_style_repair.py legacy\summaries\
Move-Item summarize_retrain_distance_repo_style.py legacy\summaries\
Move-Item summarize_rho_ablation.py legacy\summaries\
Move-Item summarize_run_fixedcrit_repo_style.py legacy\summaries\
Move-Item summarize_stage3h_h2_runtime_repair.py legacy\summaries\
Move-Item summarize_topo_repair.py legacy\summaries\
Move-Item summarize_topo_repair_all.py legacy\summaries\
```

### 5. Move table-building scripts into `legacy/tables/`

```powershell
Move-Item make_stage1_tables.py legacy\tables\
Move-Item make_stage2_final_summary.py legacy\tables\
Move-Item make_stage3_final_consolidated_summary.py legacy\tables\
Move-Item make_stage3_final_summary.py legacy\tables\
Move-Item make_stage3f_design_check.py legacy\tables\
```

### 6. Move exploratory modules into `legacy/modules/`

These are useful for audit history, but should not remain in the mainline module set unless you explicitly still depend on them.

```powershell
Move-Item utils\fed_affine_analytic.py legacy\modules\
Move-Item utils\fed_bus_client.py legacy\modules\
Move-Item utils\fed_linear_runtime.py legacy\modules\
Move-Item utils\fed_partial_input.py legacy\modules\
Move-Item utils\fed_tamu_pipeline.py legacy\modules\
Move-Item utils\fed_tamu_simulator.py legacy\modules\
Move-Item utils\physics_safe_reweight.py legacy\modules\
Move-Item utils\retrain_metrics.py legacy\modules\
```

Keep these in `utils/`:

- `dataset.py`
- `fed_block_influence.py`
- `fed_cache_utils.py`
- `fed_client_runtime.py`
- `fed_data_utils.py`
- `fed_influence_utils.py`
- `fed_local_head.py`
- `fed_repair_utils.py`
- `fed_secure_aggregation.py`
- `fed_server_runtime.py`
- `fed_topology_nn.py`
- `fed_vjp_utils.py`
- `funcs.py`
- `index_utils.py`
- `linear_performance.py`
- `linear_reg.py`
- `net.py`
- `optimization.py`
- `reweight_utils.py`
- `topology.py`
- `topo_affine.py`
- `topo_unlearn.py`
- `utils.py`
- `__init__.py`

If `physics_safe_reweight.py` is still used by the formal method, keep it in `utils/` instead of moving it.

### 7. Move tests into `legacy/tests/`

```powershell
Move-Item test_index_utils.py legacy\tests\
Move-Item test_reweight_utils.py legacy\tests\
Move-Item test_topo_affine.py legacy\tests\
Move-Item test_topo_unlearn.py legacy\tests\
Move-Item test_topology.py legacy\tests\
```

If you want an active test suite, later reintroduce a small clean `tests/` directory with only currently supported tests.

### 8. Move project-history docs into `legacy/docs/`

```powershell
Move-Item CG_SCAN_PLAN.md legacy\docs\
Move-Item OPTIMIZATION_ROADMAP.md legacy\docs\
Move-Item PROJECT_MEMORY.md legacy\docs\
```

## Artifacts Move Plan

These should not stay mixed with source code.

### 1. Move archived experiment trees

```powershell
Move-Item archive artifacts\
```

### 2. Move report deliverables

```powershell
Move-Item deliverables artifacts\
```

### 3. Move runtime outputs

```powershell
Move-Item outputs artifacts\
Move-Item simulation_result artifacts\
```

### 4. Move temporary files and extracted texts

Move these into `artifacts\tmp\`:

```powershell
Move-Item tmp_* artifacts\tmp\
Move-Item tmp_linear_ocr artifacts\tmp\
Move-Item stage3f_decision_table.csv artifacts\tmp\
Move-Item analysis_fed_visualization.py artifacts\tmp\
```

If `analysis_fed_visualization.py` is still actively used for figures, place it in `legacy/tables/` instead.

### 5. Move local paper copies and derived text

```powershell
Move-Item paper_linear.pdf artifacts\papers\
Move-Item paper_tamu.pdf artifacts\papers\
Move-Item paper_tamu_extract.txt artifacts\papers\
Move-Item paper_tamu_snippets.txt artifacts\papers\
Move-Item tmp_user_paper_extract.txt artifacts\papers\
```

## Direct Delete Plan

These should be deleted instead of archived unless you have a specific reason to keep them.

### 1. Delete duplicate root module

Delete:

- `fed_local_head.py`

Reason:

- `utils/fed_local_head.py` is the real maintained version
- the root copy creates ambiguity

Suggested command:

```powershell
Remove-Item fed_local_head.py
```

### 2. Delete cache / IDE folders

```powershell
Remove-Item -Recurse -Force .ipynb_checkpoints
Remove-Item -Recurse -Force .idea
Remove-Item -Recurse -Force __pycache__
Remove-Item -Recurse -Force utils\__pycache__
```

Do not delete `anaconda_projects/` unless you have checked that nothing in the repo still relies on it.

## Import Fix Checklist

After the moves above, update these import patterns.

### Root scripts moved to `core/`

Old:

```python
from func_operation import ...
```

New:

```python
from core.func_operation import ...
```

Or, if you prefer simple execution from repo root, convert `core/` into a package by adding:

- `core/__init__.py`

and run scripts as modules:

```powershell
python -m core.eval_fed_unchange ...
python -m core.eval_fed_unlearn ...
```

### Paths in Hydra entrypoints

If scripts move into `core/`, this decorator still works only if the relative config path is updated correctly:

Old:

```python
@hydra.main(version_base=None, config_path="conf", config_name="config")
```

New option A:

```python
@hydra.main(version_base=None, config_path="../conf", config_name="config")
```

New option B:

Use an absolute resolved config path pattern instead of a fragile relative string.

### Legacy scripts

Once moved, legacy scripts do not need perfect polish if they are not part of the active mainline. They only need enough path fixes to remain runnable on demand.

## `.gitignore` Update Plan

Add these lines if you want the cleaned repo to stay clean:

```gitignore
.idea/
.ipynb_checkpoints/
__pycache__/
legacy/
artifacts/outputs/
artifacts/simulation_result/
artifacts/tmp/
artifacts/deliverables/
*.png
```

Do not ignore all of `artifacts/` if you plan to keep archived experiment manifests or selected reports in version control.

## Recommended Execution Order

Run cleanup in this order:

1. Create target directories.
2. Delete `fed_local_head.py`.
3. Move mainline scripts to `core/`.
4. Add `core/__init__.py`.
5. Fix imports and Hydra config paths in active mainline scripts.
6. Move legacy entrypoints, runners, summaries, tables, tests, and docs.
7. Move non-code artifacts.
8. Delete cache and IDE folders.
9. Update `readme.md` to reference new mainline commands.
10. Run a smoke test:
   - `python -m core.eval_fed_unlearn ...`
   - `python -m core.eval_fed_unchange ...`

## Mainline Smoke Test

After cleanup, these commands should work:

```powershell
python -m core.train_nn model=conv
python -m core.gen_index model=conv
python -m core.eval_fed_unlearn model=conv unlearn_prop=0.2
python -m core.eval_fed_unchange model=conv unlearn_prop=0.2 +index_mode=helpful +index_criteria=cost +fed_mode=block_Hk
```

## Minimal Mainline Manifest

If you want the shortest defendable paper-code release, the active code should reduce to:

- `core/clean_data.py`
- `core/train_nn.py`
- `core/gen_index.py`
- `core/gen_event_index.py`
- `core/eval_fed_unlearn.py`
- `core/eval_fed_unchange.py`
- `core/func_operation.py`
- `conf/**`
- selected `utils/**`
- `data/`
- `torch_influence/`
- `readme.md`

Everything else can be legacy or artifact material.
