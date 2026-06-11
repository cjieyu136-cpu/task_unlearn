"""
eval_fed_cg_probe.py

Fast CG/influence probe for federated runtime experiments.

Purpose
-------
This script is a lightweight alternative to ``eval_fed_unchange.py`` when we
only want to compare solver-health / alignment behavior across CG settings.

It intentionally stops after:
    - global grad / IHVP / score
    - federated grad / IHVP / score
    - alignment + CG diagnostics

It does NOT run the full repair grid, so it is much cheaper than the full
unchange evaluation.
"""

from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from utils.dataset import NewDataset
from utils.fed_data_utils import (
    build_result_dir,
    describe_context,
    prepare_repo_affine_context,
)
from utils.index_utils import load_and_split_unlearn_datasets
from eval_fed_unchange import (
    compute_runtime_block_influence_for_eval,
)


def _clone_dataset_with_index(dataset: Any, index: np.ndarray) -> NewDataset:
    feature = dataset.feature[index]
    target = dataset.target[index]
    cloned = NewDataset(feature, target, dataset.target_mean, dataset.target_std)
    if hasattr(dataset, "is_scale"):
        cloned.is_scale = bool(dataset.is_scale)
    return cloned


def _uniform_subset_index(n: int, limit: int) -> np.ndarray:
    if limit <= 0 or limit >= n:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, num=limit, dtype=int))


def subset_dataset_uniform(dataset: Any, limit: int) -> Any:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    index = _uniform_subset_index(len(dataset), int(limit))
    return _clone_dataset_with_index(dataset, index)


def build_probe_summary_row(cfg, ctx, influence, dataset_score, dataset_remain, dataset_test_affine, result_dir):
    common = ctx["common"]
    runtime_info = influence["extra_info"].get("runtime_info", {})
    topology_info = influence["extra_info"].get("topology_extra_info", {})
    probe_index_mode = str(OmegaConf.select(cfg, "probe_index_mode_effective", default=common["index_mode"]))

    return {
        "model_type": common["model_type"],
        "eval_kind": "cg_probe",
        "result_tag": str(OmegaConf.select(cfg, "result_tag", default="")),
        "feature_mode": str(OmegaConf.select(cfg, "feature_mode", default="precomputed_local_cache")),
        "secure_agg_mode": str(OmegaConf.select(cfg, "secure_agg_mode", default="none")),
        "requested_index_mode": str(common["index_mode"]),
        "probe_index_mode": probe_index_mode,
        "probe_train_size": int(len(dataset_score)),
        "probe_remain_size": int(len(dataset_remain)),
        "probe_test_size": int(len(dataset_test_affine)),
        "damping": float(OmegaConf.select(cfg, "damping", default=np.nan)),
        "block_damping": float(common["block_damping"]),
        "cg_tol": float(OmegaConf.select(cfg, "cg_tol", default=np.nan)),
        "cg_maxiter": int(OmegaConf.select(cfg, "cg_maxiter", default=-1)),
        "gnh": bool(OmegaConf.select(cfg, "gnh", default=False)),
        "num_bus_clients": int(common["num_bus_clients"]),
        "bus_groups": influence["bus_groups"],
        "topology_partition_mode": topology_info.get("topology_partition_mode", ""),
        "topology_tau": topology_info.get("topology_tau", np.nan),
        "fusion_topology_alpha": topology_info.get("fusion_topology_alpha", np.nan),
        "encoder_self_weight": topology_info.get("encoder_self_weight", np.nan),
        "encoder_topology_mode": topology_info.get("encoder_topology_mode", ""),
        "fusion_feature_dim": topology_info.get("fusion_feature_dim", np.nan),
        "grad_cosine_similarity": influence["grad_alignment"]["cosine_similarity"],
        "grad_relative_l2_error": influence["grad_alignment"]["relative_l2_error"],
        "M_cosine_similarity": influence["M_alignment"]["cosine_similarity"],
        "M_relative_l2_error": influence["M_alignment"]["relative_l2_error"],
        "score_cosine_similarity": influence["score_alignment"]["cosine_similarity"],
        "score_relative_l2_error": influence["score_alignment"]["relative_l2_error"],
        "global_cg_info": influence["hvp_info_global"].get("cg_info", np.nan),
        "global_cg_iterations": influence["hvp_info_global"].get("cg_iterations", np.nan),
        "global_cg_relative_residual": influence["hvp_info_global"].get("cg_relative_residual", np.nan),
        "fed_cg_info": influence["hvp_info_fed"].get("cg_info", np.nan),
        "fed_cg_iterations": influence["hvp_info_fed"].get("cg_iterations", np.nan),
        "fed_cg_relative_residual": influence["hvp_info_fed"].get("cg_relative_residual", np.nan),
        "H_local_condition": runtime_info.get("H_condition", np.nan),
        "H_local_solver": runtime_info.get("solver", ""),
        "runtime_payload_total_elements": runtime_info.get("payload_total_elements", np.nan),
        "result_dir": str(result_dir),
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("========== eval_fed_cg_probe: Fast runtime CG probe ==========")

    ctx = prepare_repo_affine_context(cfg)
    common = ctx["common"]
    feature_mode = str(OmegaConf.select(cfg, "feature_mode", default="precomputed_local_cache"))
    secure_agg_mode = str(OmegaConf.select(cfg, "secure_agg_mode", default="none"))

    probe_train_size = int(OmegaConf.select(cfg, "probe_train_size", default=512))
    probe_test_size = int(OmegaConf.select(cfg, "probe_test_size", default=128))

    print("---------- context ----------")
    for k, v in describe_context(ctx).items():
        print(f"{k}: {v}")
    print("feature_mode:", feature_mode)
    print("secure_agg_mode:", secure_agg_mode)
    print("probe_train_size:", probe_train_size)
    print("probe_test_size:", probe_test_size)

    base_dir = build_result_dir(
        cfg=cfg,
        model_type=common["model_type"],
        fed_mode=common["fed_mode"],
        eval_kind="unlearn",
        index_mode=common["index_mode"],
        index_criteria=common["index_criteria"],
    ).resolve()

    result_dir = (
        base_dir.parent.parent
        / "cg_probe_runtime"
        / feature_mode
        / f"secure_{secure_agg_mode}"
        / base_dir.name
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    print("result_dir:", result_dir)

    dataset_train_affine = subset_dataset_uniform(ctx["dataset_train_affine"], probe_train_size)
    dataset_test_affine = subset_dataset_uniform(ctx["dataset_test_affine"], probe_test_size)

    requested_index_mode = str(common["index_mode"]).lower()
    effective_probe_index_mode = requested_index_mode
    if len(dataset_train_affine) < len(ctx["dataset_train_affine"]) and requested_index_mode != "random":
        effective_probe_index_mode = "random"
        OmegaConf.update(cfg, "probe_index_mode_effective", effective_probe_index_mode, force_add=True)
        print(
            "probe index_mode fallback:",
            f"{requested_index_mode} -> {effective_probe_index_mode}",
            "(subset probe avoids full-dataset precomputed index mismatch)",
        )
    else:
        OmegaConf.update(cfg, "probe_index_mode_effective", effective_probe_index_mode, force_add=True)

    dataset_unlearn, dataset_remain, unlearn_obj = load_and_split_unlearn_datasets(
        cfg=cfg,
        model_type=common["model_type"],
        dataset=dataset_train_affine,
        index_mode=effective_probe_index_mode,
        index_criteria=common["index_criteria"],
        unlearn_prop=float(cfg.unlearn_prop),
        selection_policy=common["selection_policy"],
        candidate_ratio=common["candidate_ratio"],
    )

    print("probe train rows:", len(dataset_train_affine))
    print("probe remain rows:", len(dataset_remain))
    print("probe unlearn rows:", len(dataset_unlearn))
    print("probe test rows:", len(dataset_test_affine))

    influence = compute_runtime_block_influence_for_eval(
        cfg=cfg,
        ctx=ctx,
        dataset_test_affine=dataset_test_affine,
        dataset_remain=dataset_remain,
        dataset_score=dataset_train_affine,
        unlearn_obj=unlearn_obj,
        result_dir=result_dir,
    )

    if bool(OmegaConf.select(cfg, "probe_save_arrays", default=False)):
        np.save(result_dir / "grad_global.npy", influence["grad_global"])
        np.save(result_dir / "grad_fed.npy", influence["grad_fed"])
        np.save(result_dir / "M_global.npy", influence["M_global"])
        np.save(result_dir / "M_fed.npy", influence["M_fed"])
        np.save(result_dir / "scores_global.npy", influence["scores_global"])
        np.save(result_dir / "scores_fed.npy", influence["scores_fed"])

    print("---------- probe alignment ----------")
    print("grad:", influence["grad_alignment"])
    print("M:", influence["M_alignment"])
    print("score:", influence["score_alignment"])

    summary_row = build_probe_summary_row(
        cfg=cfg,
        ctx=ctx,
        influence=influence,
        dataset_score=dataset_train_affine,
        dataset_remain=dataset_remain,
        dataset_test_affine=dataset_test_affine,
        result_dir=result_dir,
    )
    summary_df = pd.DataFrame([summary_row])
    summary_path = result_dir / "fed_cg_probe_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    log = {
        "context": describe_context(ctx),
        "probe_train_size": int(len(dataset_train_affine)),
        "probe_remain_size": int(len(dataset_remain)),
        "probe_test_size": int(len(dataset_test_affine)),
        "grad_alignment": influence["grad_alignment"],
        "M_alignment": influence["M_alignment"],
        "score_alignment": influence["score_alignment"],
        "hvp_info_global": influence["hvp_info_global"],
        "hvp_info_fed": influence["hvp_info_fed"],
        "score_info_global": influence["score_info_global"],
        "score_info_fed": influence["score_info_fed"],
        "extra_info": influence["extra_info"],
    }
    np.save(result_dir / "fed_cg_probe_log.npy", log, allow_pickle=True)

    print("\n========== Done ==========")
    print("summary:", summary_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
