"""
utils/fed_tamu_pipeline.py

Repository-style Fed-TA-MU pipeline wrapper.

Purpose
-------
This module turns the already validated Stage-3 runners into a reusable
pipeline that resembles the original repository architecture, without modifying
the original entrypoints `eval_unlearn.py` and `eval_unchange.py`.

Main entrypoints using this module:
    eval_fed_unlearn.py
    eval_fed_unchange.py

Default strongest mode:
    fed_mode=block_Hk

Supported Fed modes in this first integration:
    block_Hk
        Stage 3F client-local theta_k + client-local Hessian H_k + block score.
        This is the main integrated path.

    local_head
        Stage 3E client-local theta_k cost gradient + repo-style IHVP/score.

    bus_group
        Stage 3D bus-group shared-theta cost gradient + repo-style IHVP/score.

Notes
-----
This file intentionally does not patch or replace original repo scripts.
It follows the repo data/model/index/evaluate conventions and writes summaries
under simulation_result/<model_type>/top_fedtamu/fed_pipeline/.

Important boundary
------------------
The final repair evaluation remains repo-compatible so that the Fed variants can
be compared directly against the original centralized TA-MU path. This is not a
production secure aggregation or asynchronous deployment implementation.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from utils import return_dataset, evaluate
from utils.optimization import Operator
from utils.topology import build_load_bus_adjacency, build_laplacian
from utils.topo_affine import return_topology_affine_model

from func_operation import (
    return_core_datasets,
    return_dataset_for_nn_affine,
)

from utils.index_utils import (
    load_and_split_unlearn_datasets,
    save_unlearn_object,
)

from utils.reweight_utils import (
    get_model_parameter_vector,
    compute_inverse_hvp_vector,
    compute_sample_scores,
    solve_reweight_problem,
    repo_style_complete_unlearning,
    repo_style_repair_from_eps,
)

from utils.fed_vjp_utils import (
    centralized_cost_gradient_repo_style,
    alignment_metrics,
)

from utils.fed_bus_client import compare_bus_fed_to_central_gradient
from utils.fed_local_head import local_head_cost_gradient_model_order
from utils.fed_block_influence import (
    block_influence_audit,
    canonical_vec_to_model_order,
)


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
def format_float_for_path(x):
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def get_l1_constraints(cfg):
    root = OmegaConf.select(cfg, "l1_constraints", default=None)
    if root is not None:
        if isinstance(root, (float, int)):
            return [float(root)]
        return [float(v) for v in list(root)]

    model_level = OmegaConf.select(cfg, "model.l1_constraints", default=None)
    if model_level is not None:
        if isinstance(model_level, (float, int)):
            return [float(model_level)]
        return [float(v) for v in list(model_level)]

    return [0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.0]


def evaluate_all(model, dataset_collection, cfg):
    return {
        "mse": evaluate(model, dataset_collection, loss="mse", case_config=cfg.case),
        "mape": evaluate(model, dataset_collection, loss="mape", case_config=cfg.case),
        "cost": evaluate(model, dataset_collection, loss="cost", case_config=cfg.case),
    }


def flatten_metrics(prefix, metrics):
    row = {}
    for loss_name, split_dict in metrics.items():
        for split_name, value in split_dict.items():
            row[f"{prefix}_{loss_name}_{split_name}"] = float(value)
    return row


def build_row(method, l1_constraint, metrics, base_info, eps_info=None, repair_info=None, diff_info=None):
    row = dict(base_info)
    row["method"] = method
    row["l1_constraint"] = np.nan if l1_constraint is None else float(l1_constraint)
    row.update(flatten_metrics("metric", metrics))

    if eps_info:
        for k in ["eps_l1", "eps_linf", "eps_min", "eps_max", "solver", "status"]:
            row[k] = eps_info.get(k, np.nan)

    if repair_info:
        row["weighted_ihvp_norm"] = repair_info.get("weighted_ihvp_norm", np.nan)
        row["parameter_repair_norm"] = repair_info.get("parameter_repair_norm", np.nan)

    if diff_info:
        row.update(diff_info)

    return row


def resolve_common_cfg(cfg):
    return {
        "model_type": str(cfg.model.type),
        "index_mode": OmegaConf.select(cfg, "index_mode", default=cfg.unlearn_mode),
        "index_criteria": str(OmegaConf.select(cfg, "index_criteria", default=cfg.criteria)),
        "selection_policy": OmegaConf.select(cfg, "selection_policy", default=None),
        "candidate_ratio": OmegaConf.select(cfg, "candidate_ratio", default=None),
        "rho": float(OmegaConf.select(cfg, "rho", default=1e-3)),
        "damping": float(OmegaConf.select(cfg, "damping", default=1e-8)),
        "block_damping": float(OmegaConf.select(cfg, "block_damping", default=1e-8)),
        "train_loss": str(OmegaConf.select(cfg, "train_loss", default="mse")),
        "linf_constraint": float(OmegaConf.select(cfg, "linf_constraint", default=1.0)),
        "batch_size": int(OmegaConf.select(cfg, "fed_pipeline_batch_size", default=128)),
        "num_bus_clients": int(OmegaConf.select(cfg, "num_bus_clients", default=4)),
        "bus_groups": OmegaConf.select(cfg, "bus_groups", default=None),
        "fed_mode": str(OmegaConf.select(cfg, "fed_mode", default="block_Hk")),
    }


def make_result_dir(cfg, model_type, fed_mode, eval_kind, index_mode=None, index_criteria=None):
    if eval_kind == "unlearn":
        short = (
            f"{str(index_mode).lower()}_{index_criteria}"
            f"_p{format_float_for_path(float(cfg.unlearn_prop))}"
            f"_r{format_float_for_path(float(OmegaConf.select(cfg, 'rho', default=1e-3)))}"
            f"_bc{int(OmegaConf.select(cfg, 'num_bus_clients', default=4))}"
        )
    else:
        short = (
            f"unchange"
            f"_r{format_float_for_path(float(OmegaConf.select(cfg, 'rho', default=1e-3)))}"
            f"_bc{int(OmegaConf.select(cfg, 'num_bus_clients', default=4))}"
        )

    return (
        Path(str(cfg.simulation_dir))
        / model_type
        / "top_fedtamu"
        / "fed_pipeline"
        / str(fed_mode)
        / eval_kind
        / short
    )


# ---------------------------------------------------------------------
# Data/model preparation
# ---------------------------------------------------------------------
def prepare_affine_context(cfg) -> Dict[str, Any]:
    common = resolve_common_cfg(cfg)
    model_type = common["model_type"]

    if "nn" not in model_type:
        raise ValueError("Fed-TA-MU pipeline currently supports nn affine-head setting only.")

    dataset_train, dataset_test = return_dataset(cfg)

    dataset_core, dataset_sensitive = return_core_datasets(
        cfg,
        dataset_to_be_split=dataset_train,
    )

    dataset_train_affine, dataset_test_affine = return_dataset_for_nn_affine(
        cfg,
        dataset_sensitive,
        dataset_test,
    )

    operator = Operator(case_config=cfg.case)
    A_grid = build_load_bus_adjacency(operator)
    L_grid = build_laplacian(A_grid)

    model_ori, parameter_ori_raw = return_topology_affine_model(
        dataset=dataset_train_affine,
        L_grid=L_grid,
        rho=common["rho"],
        damping=common["damping"],
    )
    model_ori.eval()

    parameter_ori = get_model_parameter_vector(model_ori)
    parameter_ori_raw = np.asarray(parameter_ori_raw, dtype=float).reshape(-1)

    return {
        "common": common,
        "dataset_train": dataset_train,
        "dataset_test": dataset_test,
        "dataset_core": dataset_core,
        "dataset_sensitive": dataset_sensitive,
        "dataset_train_affine": dataset_train_affine,
        "dataset_test_affine": dataset_test_affine,
        "operator": operator,
        "A_grid": A_grid,
        "L_grid": L_grid,
        "model_ori": model_ori,
        "parameter_ori": parameter_ori,
        "parameter_ori_raw": parameter_ori_raw,
        "raw_parameter_diff": float(np.linalg.norm(parameter_ori - parameter_ori_raw)),
    }


# ---------------------------------------------------------------------
# Influence computation modes
# ---------------------------------------------------------------------
def compute_fed_influence(
    cfg,
    ctx: Dict[str, Any],
    dataset_test_affine,
    dataset_remain,
    dataset_train_affine,
) -> Dict[str, Any]:
    """
    Compute Fed influence quantities according to cfg.fed_mode.

    Returns:
        grad_global
        grad_fed
        M_global
        M_fed
        scores_global
        scores_fed
        alignment dicts
        mode-specific client summaries
    """
    common = ctx["common"]
    model_ori = ctx["model_ori"]
    batch_size = common["batch_size"]
    train_loss = common["train_loss"]

    loader_train = DataLoader(dataset_train_affine, batch_size=batch_size, shuffle=False)
    loader_remain = DataLoader(dataset_remain, batch_size=batch_size, shuffle=False)

    grad_global = centralized_cost_gradient_repo_style(
        cfg=cfg,
        model=model_ori,
        dataset=dataset_test_affine,
        batch_size=batch_size,
    )

    # Global baseline influence.
    M_global, hvp_info_global = compute_inverse_hvp_vector(
        cfg=cfg,
        model_train=model_ori,
        loader_train=loader_remain,
        vec=grad_global,
        train_loss=train_loss,
        sign=-1.0,
    )

    scores_global, score_info_global = compute_sample_scores(
        cfg=cfg,
        model_train=model_ori,
        loader_hessian=loader_remain,
        loader_score=loader_train,
        dataset_score=dataset_train_affine,
        M_vec=M_global,
        train_loss=train_loss,
        normalize_by_n=True,
    )

    fed_mode = common["fed_mode"]

    client_tables = {}

    if fed_mode == "bus_group":
        fed = compare_bus_fed_to_central_gradient(
            cfg=cfg,
            model=model_ori,
            dataset=dataset_test_affine,
            num_clients=common["num_bus_clients"],
            bus_groups=common["bus_groups"],
            batch_size=batch_size,
        )

        grad_fed = fed["grad_fed_bus"]
        grad_alignment = alignment_metrics(grad_global, grad_fed)

        M_fed, hvp_info_fed = compute_inverse_hvp_vector(
            cfg=cfg,
            model_train=model_ori,
            loader_train=loader_remain,
            vec=grad_fed,
            train_loss=train_loss,
            sign=-1.0,
        )

        scores_fed, score_info_fed = compute_sample_scores(
            cfg=cfg,
            model_train=model_ori,
            loader_hessian=loader_remain,
            loader_score=loader_train,
            dataset_score=dataset_train_affine,
            M_vec=M_fed,
            train_loss=train_loss,
            normalize_by_n=True,
        )

        client_tables["bus_client_payload"] = pd.DataFrame(fed["per_client"])
        bus_groups_string = fed["bus_groups_string"]
        extra_info = {"fed_info": fed["fed_info"]}

    elif fed_mode == "local_head":
        fed = local_head_cost_gradient_model_order(
            cfg=cfg,
            global_model=model_ori,
            dataset=dataset_test_affine,
            num_clients=common["num_bus_clients"],
            bus_groups=common["bus_groups"],
            batch_size=batch_size,
        )

        grad_fed = fed["local_grad_model_order"]
        grad_alignment = alignment_metrics(grad_global, grad_fed)

        M_fed, hvp_info_fed = compute_inverse_hvp_vector(
            cfg=cfg,
            model_train=model_ori,
            loader_train=loader_remain,
            vec=grad_fed,
            train_loss=train_loss,
            sign=-1.0,
        )

        scores_fed, score_info_fed = compute_sample_scores(
            cfg=cfg,
            model_train=model_ori,
            loader_hessian=loader_remain,
            loader_score=loader_train,
            dataset_score=dataset_train_affine,
            M_vec=M_fed,
            train_loss=train_loss,
            normalize_by_n=True,
        )

        client_tables["local_head_client_payload"] = pd.DataFrame(fed["per_client"])
        bus_groups_string = fed["bus_groups_string"]
        extra_info = {
            "prediction_alignment": fed["prediction_alignment"],
            "canonical_vjp_alignment": fed["vjp_alignment"],
            "model_order_alignment": fed["model_order_alignment"],
            "local_head_info": fed["info"],
        }

    elif fed_mode == "block_Hk":
        block = block_influence_audit(
            cfg=cfg,
            model=model_ori,
            dataset_test=dataset_test_affine,
            dataset_remain=dataset_remain,
            dataset_score=dataset_train_affine,
            M_global_model_order=M_global,
            scores_global=scores_global,
            num_clients=common["num_bus_clients"],
            bus_groups=common["bus_groups"],
            batch_size=batch_size,
            block_damping=common["block_damping"],
        )

        feature_dim = int(dataset_train_affine.feature.shape[1])
        output_dim = int(dataset_train_affine.target.shape[1])
        grad_fed = canonical_vec_to_model_order(
            model=model_ori,
            feature_dim=feature_dim,
            output_dim=output_dim,
            canonical_vec=block["grad_result"]["grad_canonical"],
        )

        grad_alignment = alignment_metrics(grad_global, grad_fed)
        M_fed = block["M_block_model_order"]
        scores_fed = block["scores_block"]["scores"]

        hvp_info_fed = block["M_block"]["info"]
        score_info_fed = block["scores_block"]["info"]

        client_tables["block_grad_client_summary"] = pd.DataFrame(block["grad_result"]["per_client"])
        client_tables["block_M_client_summary"] = pd.DataFrame(block["M_block"]["per_client"])
        client_tables["block_score_client_summary"] = pd.DataFrame(block["scores_block"]["per_client"])

        bus_groups_string = block["info"]["bus_groups"]
        extra_info = {
            "block_info": block["info"],
            "M_block_info": block["M_block"]["info"],
        }

    else:
        raise ValueError(
            f"Unsupported fed_mode={fed_mode}. Supported: bus_group, local_head, block_Hk"
        )

    M_alignment = alignment_metrics(M_global, M_fed)
    score_alignment = alignment_metrics(scores_global, scores_fed)

    return {
        "fed_mode": fed_mode,
        "grad_global": grad_global,
        "grad_fed": grad_fed,
        "M_global": M_global,
        "M_fed": M_fed,
        "scores_global": scores_global,
        "scores_fed": scores_fed,
        "grad_alignment": grad_alignment,
        "M_alignment": M_alignment,
        "score_alignment": score_alignment,
        "hvp_info_global": hvp_info_global,
        "hvp_info_fed": hvp_info_fed,
        "score_info_global": score_info_global,
        "score_info_fed": score_info_fed,
        "client_tables": client_tables,
        "bus_groups": bus_groups_string,
        "extra_info": extra_info,
    }


# ---------------------------------------------------------------------
# Fed unlearning pipeline
# ---------------------------------------------------------------------
def run_fed_unlearn_pipeline(cfg) -> Dict[str, Any]:
    """
    Main Fed-TA-MU unlearning pipeline.

    This is the integrated repo-style entrypoint used by eval_fed_unlearn.py.
    """
    ctx = prepare_affine_context(cfg)
    common = ctx["common"]
    model_type = common["model_type"]

    dataset_train_affine = ctx["dataset_train_affine"]
    dataset_test_affine = ctx["dataset_test_affine"]
    model_ori = ctx["model_ori"]
    parameter_ori = ctx["parameter_ori"]

    result_dir = make_result_dir(
        cfg,
        model_type=model_type,
        fed_mode=common["fed_mode"],
        eval_kind="unlearn",
        index_mode=common["index_mode"],
        index_criteria=common["index_criteria"],
    ).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset_unlearn, dataset_remain, unlearn_obj = load_and_split_unlearn_datasets(
        cfg=cfg,
        model_type=model_type,
        dataset=dataset_train_affine,
        index_mode=common["index_mode"],
        index_criteria=common["index_criteria"],
        unlearn_prop=float(cfg.unlearn_prop),
        selection_policy=common["selection_policy"],
        candidate_ratio=common["candidate_ratio"],
    )

    effective_policy = unlearn_obj["metadata"].get("selection_policy", "event_or_none")
    save_unlearn_object(unlearn_obj, str(result_dir))

    loader_remain = DataLoader(dataset_remain, batch_size=common["batch_size"], shuffle=False)

    influence = compute_fed_influence(
        cfg=cfg,
        ctx=ctx,
        dataset_test_affine=dataset_test_affine,
        dataset_remain=dataset_remain,
        dataset_train_affine=dataset_train_affine,
    )

    # Save influence arrays and client summaries.
    np.save(result_dir / "grad_global.npy", influence["grad_global"])
    np.save(result_dir / "grad_fed.npy", influence["grad_fed"])
    np.save(result_dir / "M_global.npy", influence["M_global"])
    np.save(result_dir / "M_fed.npy", influence["M_fed"])
    np.save(result_dir / "scores_global.npy", influence["scores_global"])
    np.save(result_dir / "scores_fed.npy", influence["scores_fed"])

    for name, df in influence["client_tables"].items():
        df.to_csv(result_dir / f"{name}.csv", index=False)

    # Complete unlearning baseline.
    parameter_complete, model_complete, complete_info = repo_style_complete_unlearning(
        cfg=cfg,
        model_original=model_ori,
        loader_remain=loader_remain,
        dataset_remain=dataset_remain,
        parameter_original=parameter_ori,
        train_loss=common["train_loss"],
    )

    dataset_collection = {
        "remain": dataset_remain,
        "unlearn": dataset_unlearn,
        "test": dataset_test_affine,
    }

    metrics_original = evaluate_all(model_ori, dataset_collection, cfg)
    metrics_complete = evaluate_all(model_complete, dataset_collection, cfg)

    base_info = {
        "model_type": model_type,
        "eval_kind": "unlearn",
        "fed_mode": common["fed_mode"],
        "index_mode": str(common["index_mode"]),
        "index_criteria": str(common["index_criteria"]),
        "repair_criteria": "cost",
        "selection_policy": str(effective_policy),
        "unlearn_prop": float(cfg.unlearn_prop),
        "rho": float(common["rho"]),
        "block_damping": float(common["block_damping"]),
        "linf_constraint": float(common["linf_constraint"]),
        "num_bus_clients": int(common["num_bus_clients"]),
        "bus_groups": influence["bus_groups"],
        "raw_parameter_diff": float(ctx["raw_parameter_diff"]),
        "grad_cosine_similarity": influence["grad_alignment"]["cosine_similarity"],
        "grad_relative_l2_error": influence["grad_alignment"]["relative_l2_error"],
        "M_cosine_similarity": influence["M_alignment"]["cosine_similarity"],
        "M_relative_l2_error": influence["M_alignment"]["relative_l2_error"],
        "score_cosine_similarity": influence["score_alignment"]["cosine_similarity"],
        "score_relative_l2_error": influence["score_alignment"]["relative_l2_error"],
    }

    # Optional mode-specific details.
    if common["fed_mode"] == "local_head":
        pa = influence["extra_info"].get("prediction_alignment", {})
        ca = influence["extra_info"].get("canonical_vjp_alignment", {})
        base_info.update({
            "prediction_rmse": pa.get("prediction_rmse", np.nan),
            "prediction_max_abs_diff": pa.get("prediction_max_abs_diff", np.nan),
            "canonical_vjp_cosine_similarity": ca.get("cosine_similarity", np.nan),
            "canonical_vjp_relative_l2_error": ca.get("relative_l2_error", np.nan),
        })
    elif common["fed_mode"] == "block_Hk":
        mb = influence["extra_info"].get("M_block_info", {})
        base_info.update({
            "H_local_condition": mb.get("H_condition", np.nan),
            "H_local_shape": str(mb.get("H_shape", "")),
        })

    rows = [
        build_row("original", None, metrics_original, base_info),
        build_row("complete", None, metrics_complete, base_info),
    ]

    remain_index = unlearn_obj["remain_index"]
    scores_remain_global = influence["scores_global"][remain_index]
    scores_remain_fed = influence["scores_fed"][remain_index]

    repair_logs = []

    for l1_constraint in get_l1_constraints(cfg):
        eps_global, eps_info_global = solve_reweight_problem(
            scores_remain=scores_remain_global,
            l1_constraint=float(l1_constraint),
            linf_constraint=common["linf_constraint"],
        )

        parameter_repair_global, model_repair_global, repair_info_global = repo_style_repair_from_eps(
            cfg=cfg,
            model_original=model_ori,
            dataset_remain=dataset_remain,
            eps_remain=eps_global,
            parameter_original=parameter_ori,
            batch_size=common["batch_size"],
            train_loss=common["train_loss"],
        )

        metrics_global = evaluate_all(model_repair_global, dataset_collection, cfg)

        eps_fed, eps_info_fed = solve_reweight_problem(
            scores_remain=scores_remain_fed,
            l1_constraint=float(l1_constraint),
            linf_constraint=common["linf_constraint"],
        )

        parameter_repair_fed, model_repair_fed, repair_info_fed = repo_style_repair_from_eps(
            cfg=cfg,
            model_original=model_ori,
            dataset_remain=dataset_remain,
            eps_remain=eps_fed,
            parameter_original=parameter_ori,
            batch_size=common["batch_size"],
            train_loss=common["train_loss"],
        )

        metrics_fed = evaluate_all(model_repair_fed, dataset_collection, cfg)

        param_diff = float(np.linalg.norm(parameter_repair_fed - parameter_repair_global))
        eps_diff = float(np.linalg.norm(eps_fed - eps_global))

        rows.append(
            build_row(
                "repair_global_cost",
                l1_constraint,
                metrics_global,
                base_info,
                eps_info_global,
                repair_info_global,
                {
                    "parameter_l2_diff_to_global_repair": 0.0,
                    "eps_l2_diff_to_global": 0.0,
                },
            )
        )

        rows.append(
            build_row(
                "repair_fed_cost",
                l1_constraint,
                metrics_fed,
                base_info,
                eps_info_fed,
                repair_info_fed,
                {
                    "parameter_l2_diff_to_global_repair": param_diff,
                    "eps_l2_diff_to_global": eps_diff,
                    "delta_metric_cost_test_fed_minus_global": metrics_fed["cost"]["test"] - metrics_global["cost"]["test"],
                    "delta_metric_mse_test_fed_minus_global": metrics_fed["mse"]["test"] - metrics_global["mse"]["test"],
                    "delta_metric_mape_test_fed_minus_global": metrics_fed["mape"]["test"] - metrics_global["mape"]["test"],
                    "delta_metric_cost_unlearn_fed_minus_global": metrics_fed["cost"]["unlearn"] - metrics_global["cost"]["unlearn"],
                },
            )
        )

        repair_logs.append({
            "l1_constraint": float(l1_constraint),
            "global_metrics": metrics_global,
            "fed_metrics": metrics_fed,
            "global_eps_info": eps_info_global,
            "fed_eps_info": eps_info_fed,
            "parameter_l2_diff_fed_to_global": param_diff,
            "eps_l2_diff_fed_to_global": eps_diff,
        })

    summary_df = pd.DataFrame(rows)
    summary_path = result_dir / "fed_unlearn_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    log = {
        "base_info": base_info,
        "complete_info": complete_info,
        "grad_alignment": influence["grad_alignment"],
        "M_alignment": influence["M_alignment"],
        "score_alignment": influence["score_alignment"],
        "hvp_info_global": influence["hvp_info_global"],
        "hvp_info_fed": influence["hvp_info_fed"],
        "score_info_global": influence["score_info_global"],
        "score_info_fed": influence["score_info_fed"],
        "extra_info": influence["extra_info"],
        "repair_logs": repair_logs,
    }
    np.save(result_dir / "fed_unlearn_log.npy", log, allow_pickle=True)

    return {
        "result_dir": result_dir,
        "summary_path": summary_path,
        "summary_df": summary_df,
        "log": log,
    }


# ---------------------------------------------------------------------
# Fed unchanged / diagnostic pipeline
# ---------------------------------------------------------------------
def run_fed_unchange_pipeline(cfg) -> Dict[str, Any]:
    """
    Fed diagnostic/evaluation pipeline without applying an unlearn split or
    repair.

    This is the integrated entrypoint used by eval_fed_unchange.py. It verifies
    that the selected Fed mode can compute aligned gradient/IHVP/score using the
    full affine training set as the Hessian/score dataset.

    It does not claim to replace the original repo's eval_unchange semantics;
    it is a Fed-TA-MU diagnostic counterpart.
    """
    ctx = prepare_affine_context(cfg)
    common = ctx["common"]
    model_type = common["model_type"]

    dataset_train_affine = ctx["dataset_train_affine"]
    dataset_test_affine = ctx["dataset_test_affine"]
    model_ori = ctx["model_ori"]

    result_dir = make_result_dir(
        cfg,
        model_type=model_type,
        fed_mode=common["fed_mode"],
        eval_kind="unchange",
    ).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    # Use full training affine set as remain/score set for no-unlearn diagnostic.
    influence = compute_fed_influence(
        cfg=cfg,
        ctx=ctx,
        dataset_test_affine=dataset_test_affine,
        dataset_remain=dataset_train_affine,
        dataset_train_affine=dataset_train_affine,
    )

    dataset_collection = {
        "train": dataset_train_affine,
        "test": dataset_test_affine,
    }

    metrics_original = {
        "mse": evaluate(model_ori, dataset_collection, loss="mse", case_config=cfg.case),
        "mape": evaluate(model_ori, dataset_collection, loss="mape", case_config=cfg.case),
        "cost": evaluate(model_ori, dataset_collection, loss="cost", case_config=cfg.case),
    }

    base_info = {
        "model_type": model_type,
        "eval_kind": "unchange",
        "fed_mode": common["fed_mode"],
        "rho": float(common["rho"]),
        "block_damping": float(common["block_damping"]),
        "num_bus_clients": int(common["num_bus_clients"]),
        "bus_groups": influence["bus_groups"],
        "raw_parameter_diff": float(ctx["raw_parameter_diff"]),
        "grad_cosine_similarity": influence["grad_alignment"]["cosine_similarity"],
        "grad_relative_l2_error": influence["grad_alignment"]["relative_l2_error"],
        "M_cosine_similarity": influence["M_alignment"]["cosine_similarity"],
        "M_relative_l2_error": influence["M_alignment"]["relative_l2_error"],
        "score_cosine_similarity": influence["score_alignment"]["cosine_similarity"],
        "score_relative_l2_error": influence["score_alignment"]["relative_l2_error"],
    }

    row = dict(base_info)
    row.update(flatten_metrics("metric", metrics_original))

    summary_df = pd.DataFrame([row])
    summary_path = result_dir / "fed_unchange_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    for name, df in influence["client_tables"].items():
        df.to_csv(result_dir / f"{name}.csv", index=False)

    np.save(result_dir / "grad_global.npy", influence["grad_global"])
    np.save(result_dir / "grad_fed.npy", influence["grad_fed"])
    np.save(result_dir / "M_global.npy", influence["M_global"])
    np.save(result_dir / "M_fed.npy", influence["M_fed"])
    np.save(result_dir / "scores_global.npy", influence["scores_global"])
    np.save(result_dir / "scores_fed.npy", influence["scores_fed"])

    log = {
        "base_info": base_info,
        "grad_alignment": influence["grad_alignment"],
        "M_alignment": influence["M_alignment"],
        "score_alignment": influence["score_alignment"],
        "hvp_info_global": influence["hvp_info_global"],
        "hvp_info_fed": influence["hvp_info_fed"],
        "score_info_global": influence["score_info_global"],
        "score_info_fed": influence["score_info_fed"],
        "extra_info": influence["extra_info"],
    }
    np.save(result_dir / "fed_unchange_log.npy", log, allow_pickle=True)

    return {
        "result_dir": result_dir,
        "summary_path": summary_path,
        "summary_df": summary_df,
        "log": log,
    }
