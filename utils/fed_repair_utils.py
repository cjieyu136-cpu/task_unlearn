"""
utils/fed_repair_utils.py

Fed-TA-MU repair utilities.

This module mirrors the repository's reweight/repair/evaluate responsibilities,
while keeping Fed influence computation outside in fed_influence_utils.py.

Responsibilities:
    - evaluate models
    - run complete unlearning baseline
    - solve eps for global and Fed scores
    - apply repo-compatible repair
    - build summary rows
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from utils import evaluate
from utils.funcs import evaluate_cost_safety
from utils.reweight_utils import (
    repo_style_complete_unlearning,
    repo_style_repair_from_eps,
    solve_reweight_problem,
)


def get_l1_constraints(cfg) -> List[float]:
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
        "_safety": evaluate_cost_safety(model, dataset_collection, case_config=cfg.case),
    }


def flatten_metrics(prefix, metrics):
    row = {}
    for loss_name, split_dict in metrics.items():
        if loss_name == "_safety":
            continue
        for split_name, value in split_dict.items():
            row[f"{prefix}_{loss_name}_{split_name}"] = float(value)
    return row


def build_summary_row(method, l1_constraint, metrics, base_info, eps_info=None, repair_info=None, diff_info=None):
    row = dict(base_info)
    row["method"] = method
    row["l1_constraint"] = np.nan if l1_constraint is None else float(l1_constraint)
    row.update(flatten_metrics("metric", metrics))
    safety_metrics = metrics.get("_safety", {})
    for split_name, split_metrics in safety_metrics.items():
        for metric_name, value in split_metrics.items():
            row[f"safety_{metric_name}_{split_name}"] = float(value)

    if eps_info:
        for k in ["eps_l1", "eps_linf", "eps_min", "eps_max", "solver", "status"]:
            row[k] = eps_info.get(k, np.nan)

    if repair_info:
        row["weighted_ihvp_norm"] = repair_info.get("weighted_ihvp_norm", np.nan)
        row["parameter_repair_norm"] = repair_info.get("parameter_repair_norm", np.nan)

    if diff_info:
        row.update(diff_info)

    return row


def run_complete_unlearning_baseline(cfg, model_ori, dataset_remain, parameter_ori, batch_size, train_loss):
    loader_remain = DataLoader(dataset_remain, batch_size=batch_size, shuffle=False)
    return repo_style_complete_unlearning(
        cfg=cfg,
        model_original=model_ori,
        loader_remain=loader_remain,
        dataset_remain=dataset_remain,
        parameter_original=parameter_ori,
        train_loss=train_loss,
    )


def run_fed_repair_grid(
    cfg,
    model_ori,
    parameter_ori,
    dataset_remain,
    dataset_collection,
    remain_index,
    scores_global,
    scores_fed,
    base_info,
    batch_size,
    train_loss,
    linf_constraint,
    repair_criteria: str = "cost",
    sample_weight_global=None,
    sample_weight_fed=None,
    progress_csv_path: Optional[Path] = None,
):
    """
    Run global-baseline repair and Fed repair using two score vectors.

    Returns:
        rows, repair_logs
    """
    rows = []
    repair_logs = []
    repair_criteria = str(repair_criteria).lower()
    method_global = f"repair_global_{repair_criteria}"
    method_fed = f"repair_fed_{repair_criteria}"

    scores_remain_global = scores_global[remain_index]
    scores_remain_fed = scores_fed[remain_index]
    if sample_weight_global is not None:
        sample_weight_global = np.asarray(sample_weight_global, dtype=float).reshape(-1)
        weight_remain_global = sample_weight_global[remain_index]
        scores_remain_global = scores_remain_global * weight_remain_global
    if sample_weight_fed is not None:
        sample_weight_fed = np.asarray(sample_weight_fed, dtype=float).reshape(-1)
        weight_remain_fed = sample_weight_fed[remain_index]
        scores_remain_fed = scores_remain_fed * weight_remain_fed

    progress_rows = []
    completed = set()
    if progress_csv_path is not None and Path(progress_csv_path).exists():
        existing = pd.read_csv(progress_csv_path)
        progress_rows = existing.to_dict("records")
        for row in progress_rows:
            completed.add((str(row.get("method")), float(row.get("l1_constraint"))))
            rows.append(dict(row))

    for l1_constraint in get_l1_constraints(cfg):
        if (method_global, float(l1_constraint)) in completed and (method_fed, float(l1_constraint)) in completed:
            continue

        eps_global, eps_info_global = solve_reweight_problem(
            scores_remain=scores_remain_global,
            l1_constraint=float(l1_constraint),
            linf_constraint=float(linf_constraint),
        )

        parameter_repair_global, model_repair_global, repair_info_global = repo_style_repair_from_eps(
            cfg=cfg,
            model_original=model_ori,
            dataset_remain=dataset_remain,
            eps_remain=eps_global,
            parameter_original=parameter_ori,
            batch_size=batch_size,
            train_loss=train_loss,
        )

        metrics_global = evaluate_all(model_repair_global, dataset_collection, cfg)

        eps_fed, eps_info_fed = solve_reweight_problem(
            scores_remain=scores_remain_fed,
            l1_constraint=float(l1_constraint),
            linf_constraint=float(linf_constraint),
        )

        parameter_repair_fed, model_repair_fed, repair_info_fed = repo_style_repair_from_eps(
            cfg=cfg,
            model_original=model_ori,
            dataset_remain=dataset_remain,
            eps_remain=eps_fed,
            parameter_original=parameter_ori,
            batch_size=batch_size,
            train_loss=train_loss,
        )

        metrics_fed = evaluate_all(model_repair_fed, dataset_collection, cfg)

        param_diff = float(np.linalg.norm(parameter_repair_fed - parameter_repair_global))
        eps_diff = float(np.linalg.norm(eps_fed - eps_global))

        row_global = build_summary_row(
                method_global,
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
        rows.append(row_global)

        row_fed = build_summary_row(
                method_fed,
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
        rows.append(row_fed)

        if progress_csv_path is not None:
            progress_rows.extend([row_global, row_fed])
            progress_csv_path = Path(progress_csv_path)
            progress_csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(progress_rows).to_csv(progress_csv_path, index=False)

        repair_logs.append({
            "l1_constraint": float(l1_constraint),
            "global_metrics": metrics_global,
            "fed_metrics": metrics_fed,
            "global_eps_info": eps_info_global,
            "fed_eps_info": eps_info_fed,
            "parameter_l2_diff_fed_to_global": param_diff,
            "eps_l2_diff_fed_to_global": eps_diff,
        })

    return rows, repair_logs
