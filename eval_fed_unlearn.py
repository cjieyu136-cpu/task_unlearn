"""
eval_fed_unlearn.py

Federated direct-unlearning evaluation aligned with the original eval_unlearn.py
comparison shape:

    original
    direct_unlearn
    retrain

For NN models, this script supports the topology-aware local-input federation
path via:

    +feature_mode=topology_local_fusion

The model under evaluation is still a repo-compatible affine fusion head, so
metrics remain directly comparable to the original centralized last-layer path.
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from utils.fed_data_utils import (
    prepare_repo_affine_context,
    describe_context,
    build_result_dir,
    resolve_feature_and_runtime_mode,
    compact_feature_mode_for_path,
    compact_runtime_mode_for_path,
)
from utils.fed_repair_utils import evaluate_all, build_summary_row, run_complete_unlearning_baseline
from utils.index_utils import load_and_split_unlearn_datasets, save_unlearn_object
from utils.reweight_utils import get_model_parameter_vector
from utils.topo_affine import return_topology_affine_model


def _load_existing_unlearn_summary(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _summary_has_method(df: pd.DataFrame, method: str) -> bool:
    if df.empty or "method" not in df.columns:
        return False
    return bool((df["method"].astype(str) == str(method)).any())


def _append_summary_row(path: Path, row: dict) -> pd.DataFrame:
    existing = _load_existing_unlearn_summary(path)
    out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    out.to_csv(path, index=False)
    return out


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("========== eval_fed_unlearn: Fed-TA-MU direct unlearning ==========")

    ctx = prepare_repo_affine_context(cfg)
    common = ctx["common"]
    mode_info = resolve_feature_and_runtime_mode(cfg)
    runtime_input_mode = mode_info["feature_mode"]
    affine_feature_mode = str(ctx.get("extra_info", {}).get("feature_mode", runtime_input_mode))
    runtime_mode = mode_info["runtime_mode"]

    print("---------- context ----------")
    for k, v in describe_context(ctx).items():
        print(f"{k}: {v}")
    print("affine_feature_mode:", affine_feature_mode)
    print("runtime_input_mode:", runtime_input_mode)
    print("runtime_mode:", runtime_mode)

    result_dir = build_result_dir(
        cfg=cfg,
        model_type=common["model_type"],
        fed_mode=common["fed_mode"],
        eval_kind="unlearn",
        index_mode=common["index_mode"],
        index_criteria=common["index_criteria"],
    ).resolve()

    result_dir = (
        result_dir.parent.parent
        / "fed_unlearn_runtime"
        / compact_feature_mode_for_path(affine_feature_mode)
        / f"rt_{compact_runtime_mode_for_path(runtime_mode)}"
        / result_dir.name
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    print("result_dir:", result_dir)
    summary_path = result_dir / "fed_unlearn_summary.csv"
    log_path = result_dir / "fed_unlearn_log.npy"

    dataset_train_affine = ctx["dataset_train_affine"]
    dataset_test_affine = ctx["dataset_test_affine"]
    model_ori = ctx["model_ori"]
    parameter_ori = ctx["parameter_ori"]

    dataset_unlearn, dataset_remain, unlearn_obj = load_and_split_unlearn_datasets(
        cfg=cfg,
        model_type=common["model_type"],
        dataset=dataset_train_affine,
        index_mode=common["index_mode"],
        index_criteria=common["index_criteria"],
        unlearn_prop=float(cfg.unlearn_prop),
        selection_policy=common["selection_policy"],
        candidate_ratio=common["candidate_ratio"],
    )
    save_unlearn_object(unlearn_obj, str(result_dir))

    dataset_collection = {
        "remain": dataset_remain,
        "unlearn": dataset_unlearn,
        "test": dataset_test_affine,
    }

    existing_summary = _load_existing_unlearn_summary(summary_path)

    base_info = {
        "model_type": common["model_type"],
        "eval_kind": "fed_unlearn",
        "result_tag": str(OmegaConf.select(cfg, "result_tag", default="")),
        "fed_mode": common["fed_mode"],
        "feature_mode": affine_feature_mode,
        "affine_feature_mode": affine_feature_mode,
        "runtime_input_mode": runtime_input_mode,
        "runtime_mode": runtime_mode,
        "index_mode": str(common["index_mode"]),
        "index_criteria": str(common["index_criteria"]),
        "unlearn_prop": float(cfg.unlearn_prop),
        "rho": float(common["rho"]),
        "damping": float(common["damping"]),
        "block_damping": float(common["block_damping"]),
        "num_bus_clients": int(common["num_bus_clients"]),
        "bus_groups": ctx["bus_groups_string"],
        "topology_partition_enabled": ctx.get("extra_info", {}).get("topology_partition_enabled", np.nan),
        "topology_fusion_enabled": ctx.get("extra_info", {}).get("topology_fusion_enabled", np.nan),
        "topology_repair_regularization_enabled": ctx.get("extra_info", {}).get(
            "topology_repair_regularization_enabled", np.nan
        ),
        "topology_encoder_propagation_enabled": ctx.get("extra_info", {}).get(
            "topology_encoder_propagation_enabled", np.nan
        ),
        "topology_partition_mode": ctx.get("extra_info", {}).get("topology_partition_mode", ""),
        "topology_tau": ctx.get("extra_info", {}).get("topology_tau", np.nan),
        "topology_repair_rho": ctx.get("extra_info", {}).get("topology_repair_rho", np.nan),
        "fusion_topology_alpha": ctx.get("extra_info", {}).get("fusion_topology_alpha", np.nan),
        "server_feature_layout": ctx.get("extra_info", {}).get("server_feature_layout", ""),
        "fusion_feature_layout": ctx.get("extra_info", {}).get("fusion_feature_layout", ""),
        "encoder_self_weight": ctx.get("extra_info", {}).get("encoder_self_weight", np.nan),
        "encoder_input_mode": ctx.get("extra_info", {}).get("encoder_input_mode", ""),
        "fusion_feature_dim": ctx.get("extra_info", {}).get("fusion_feature_dim", np.nan),
        "fusion_use_smoothing_residual": ctx.get("extra_info", {}).get("fusion_use_smoothing_residual", np.nan),
        "fusion_normalize_feature": ctx.get("extra_info", {}).get("fusion_normalize_feature", np.nan),
        "raw_parameter_diff": float(ctx["raw_parameter_diff"]),
    }

    if not _summary_has_method(existing_summary, "original"):
        metrics_original = evaluate_all(model_ori, dataset_collection, cfg)
        existing_summary = _append_summary_row(
            summary_path,
            build_summary_row("original", None, metrics_original, base_info),
        )

    direct_info = {}
    if not _summary_has_method(existing_summary, "direct_unlearn"):
        parameter_direct, model_direct, direct_info = run_complete_unlearning_baseline(
            cfg=cfg,
            model_ori=model_ori,
            dataset_remain=dataset_remain,
            parameter_ori=parameter_ori,
            batch_size=common["batch_size"],
            train_loss=common["train_loss"],
        )
        metrics_direct = evaluate_all(model_direct, dataset_collection, cfg)
    else:
        parameter_direct = None
        model_direct = None

    model_retrain, parameter_retrain_raw = return_topology_affine_model(
        dataset=dataset_remain,
        L_grid=ctx["L_grid"],
        rho=float(ctx.get("extra_info", {}).get("topology_repair_rho", common["rho"])),
        damping=common["damping"],
    )
    model_retrain.eval()
    parameter_retrain = get_model_parameter_vector(model_retrain)
    parameter_retrain_raw = np.asarray(parameter_retrain_raw, dtype=float).reshape(-1)
    if parameter_direct is None:
        direct_row = existing_summary[existing_summary["method"].astype(str) == "direct_unlearn"].iloc[0]
        parameter_l2_diff_to_retrain = pd.to_numeric(direct_row.get("parameter_l2_diff_to_retrain"), errors="coerce")
    else:
        existing_summary = _append_summary_row(
            summary_path,
            build_summary_row(
                "direct_unlearn",
                None,
                metrics_direct,
                base_info,
                repair_info={
                    "weighted_ihvp_norm": direct_info.get("ihvp_norm", np.nan),
                    "parameter_repair_norm": direct_info.get("parameter_complete_norm", np.nan),
                },
                diff_info={
                    "cg_info": direct_info.get("cg_info", np.nan),
                    "cg_iterations": direct_info.get("cg_iterations", np.nan),
                    "cg_residual_norm": direct_info.get("cg_residual_norm", np.nan),
                    "cg_relative_residual": direct_info.get("cg_relative_residual", np.nan),
                    "parameter_l2_diff_to_retrain": float(np.linalg.norm(parameter_direct - parameter_retrain)),
                    "parameter_l2_diff_to_original": float(np.linalg.norm(parameter_direct - parameter_ori)),
                },
            ),
        )
        parameter_l2_diff_to_retrain = float(np.linalg.norm(parameter_direct - parameter_retrain))

    if not _summary_has_method(existing_summary, "retrain"):
        metrics_retrain = evaluate_all(model_retrain, dataset_collection, cfg)
        existing_summary = _append_summary_row(
            summary_path,
            build_summary_row(
                "retrain",
                None,
                metrics_retrain,
                base_info,
                diff_info={
                    "parameter_l2_diff_to_direct_unlearn": parameter_l2_diff_to_retrain,
                    "parameter_l2_diff_to_original": float(np.linalg.norm(parameter_retrain - parameter_ori)),
                    "raw_parameter_l2_diff_to_flatten": float(np.linalg.norm(parameter_retrain - parameter_retrain_raw)),
                },
            ),
        )

    summary_df = pd.read_csv(summary_path)

    for name, df in ctx.get("extra_tables", {}).items():
        out_path = result_dir / f"runtime_{name}.csv"
        if not out_path.exists():
            df.to_csv(out_path, index=False)

    log = {
        "context": describe_context(ctx),
        "extra_info": ctx.get("extra_info", {}),
        "direct_unlearn_info": direct_info,
        "parameter_l2_diff_direct_to_retrain": float(parameter_l2_diff_to_retrain) if not pd.isna(parameter_l2_diff_to_retrain) else np.nan,
        "parameter_l2_diff_direct_to_original": (
            float(np.linalg.norm(parameter_direct - parameter_ori)) if parameter_direct is not None else np.nan
        ),
        "parameter_l2_diff_retrain_to_original": float(np.linalg.norm(parameter_retrain - parameter_ori)),
        "parameter_l2_diff_retrain_raw_to_flatten": float(np.linalg.norm(parameter_retrain - parameter_retrain_raw)),
    }
    np.save(log_path, log, allow_pickle=True)

    print("\n========== Done ==========")
    print("summary:", summary_path)
    show_cols = [
        "method",
        "metric_mse_test",
        "metric_mape_test",
        "metric_cost_test",
        "metric_mse_unlearn",
        "metric_mape_unlearn",
        "metric_cost_unlearn",
        "parameter_l2_diff_to_retrain",
        "cg_info",
        "cg_iterations",
        "cg_relative_residual",
    ]
    show_cols = [c for c in show_cols if c in summary_df.columns]
    print(summary_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
