"""
utils/retrain_metrics.py

Stage 2 utility module for TOP-FedTAMU+.

This module centralizes retrain-distance / functional forgetting-completeness
metrics that were validated in Stage 1.

Core idea
---------
In machine unlearning, the clean reference model is the model retrained only on
D_remain. For the NN-affine setting used in this project:

    CNN / Conv / Mixer feature extractor is frozen
    final affine head is refit on D_remain

This module provides:

    - fit_retrain_affine_model(...)
    - prediction_distance(...)
    - evaluate_retrain_distances(...)
    - load_repair_parameters_from_result_dir(...)

Important convention
--------------------
For repo-style unlearning / repair, parameter vectors must use the same order as
the original repository flatten_model() / reconstruct_model() utilities.
Therefore, this module always uses:

    get_model_parameter_vector(model)

which internally uses utils.flatten_model when available.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from utils import evaluate, reconstruct_model
from utils.topology import build_load_bus_adjacency, build_laplacian
from utils.optimization import Operator
from utils.topo_affine import return_topology_affine_model

from utils.reweight_utils import get_model_parameter_vector


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def flatten_metrics(prefix: str, metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    row = {}
    for loss_name, split_dict in metrics.items():
        for split_name, value in split_dict.items():
            row[f"{prefix}_{loss_name}_{split_name}"] = float(value)
    return row


def evaluate_all(model, dataset_collection, cfg):
    return {
        "mse": evaluate(model, dataset_collection, loss="mse", case_config=cfg.case),
        "mape": evaluate(model, dataset_collection, loss="mape", case_config=cfg.case),
        "cost": evaluate(model, dataset_collection, loss="cost", case_config=cfg.case),
    }


# ---------------------------------------------------------------------
# Retrain reference
# ---------------------------------------------------------------------
def fit_retrain_affine_model(
    cfg: Any,
    dataset_remain: Any,
    rho: float = 1e-3,
    damping: float = 1e-8,
) -> Tuple[torch.nn.Module, np.ndarray, Dict[str, Any]]:
    """
    Fit the topology-aware affine head on D_remain only.

    This is the functional reference for unlearning completeness.

    Returns:
        model_retrain
        parameter_retrain_flatten
        info
    """
    operator = Operator(case_config=cfg.case)
    A_grid = build_load_bus_adjacency(operator)
    L_grid = build_laplacian(A_grid)

    model_retrain, parameter_retrain_raw = return_topology_affine_model(
        dataset=dataset_remain,
        L_grid=L_grid,
        rho=float(rho),
        damping=float(damping),
    )
    model_retrain.eval()

    parameter_retrain = get_model_parameter_vector(model_retrain)
    parameter_retrain_raw = np.asarray(parameter_retrain_raw, dtype=float).reshape(-1)

    raw_diff = float(np.linalg.norm(parameter_retrain - parameter_retrain_raw))

    info = {
        "rho": float(rho),
        "damping": float(damping),
        "parameter_retrain_norm": float(np.linalg.norm(parameter_retrain)),
        "raw_parameter_norm": float(np.linalg.norm(parameter_retrain_raw)),
        "raw_parameter_diff_to_flatten": raw_diff,
    }

    return model_retrain, parameter_retrain, info


# ---------------------------------------------------------------------
# Prediction distances
# ---------------------------------------------------------------------
def predict_numpy(model: torch.nn.Module, dataset: Any) -> np.ndarray:
    """
    Return unscaled predictions, following evaluate() convention.

    If dataset.is_scale is True, prediction is transformed by:
        pred * target_std + target_mean
    """
    model.eval()
    with torch.no_grad():
        pred = model(dataset.feature)
        if isinstance(pred, (tuple, list)):
            pred = pred[-1]

    pred_np = to_numpy(pred).astype(float)

    if getattr(dataset, "is_scale", False):
        mean = to_numpy(dataset.target_mean).astype(float)
        std = to_numpy(dataset.target_std).astype(float)
        pred_np = pred_np * std + mean

    return pred_np


def prediction_distance(
    model: torch.nn.Module,
    model_retrain: torch.nn.Module,
    dataset: Any,
) -> Dict[str, float]:
    """
    Functional distance between method model and retrain model on a dataset.

    This is often more stable than raw parameter L2 for topology-affine heads,
    because parameterizations can be non-unique while predictions are identical.
    """
    pred = predict_numpy(model, dataset)
    pred_ref = predict_numpy(model_retrain, dataset)

    diff = pred - pred_ref

    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    denom = np.maximum(np.abs(pred_ref), 1e-8)
    rel_mape_percent = float(np.mean(np.abs(diff) / denom) * 100.0)

    return {
        "pred_mse_to_retrain": mse,
        "pred_rmse_to_retrain": rmse,
        "pred_mae_to_retrain": mae,
        "pred_mape_to_retrain_percent": rel_mape_percent,
    }


def add_prediction_distances(
    row: Dict[str, Any],
    model: torch.nn.Module,
    model_retrain: torch.nn.Module,
    dataset_collection: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add remain/unlearn/test functional distances to row.
    """
    for split_name, dataset in dataset_collection.items():
        dist = prediction_distance(model, model_retrain, dataset)
        for k, v in dist.items():
            row[f"{split_name}_{k}"] = v
    return row


# ---------------------------------------------------------------------
# Parameter / metric rows
# ---------------------------------------------------------------------
def make_distance_row(
    method: str,
    model: torch.nn.Module,
    parameter: np.ndarray,
    model_retrain: torch.nn.Module,
    parameter_retrain: np.ndarray,
    dataset_collection: Dict[str, Any],
    cfg: Any,
    base_info: Optional[Dict[str, Any]] = None,
    l1_constraint: Optional[float] = None,
    eps_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create one row containing:
        - normal metrics on remain/unlearn/test
        - parameter distance to retrain
        - prediction distance to retrain
        - optional eps info
    """
    parameter = np.asarray(parameter, dtype=float).reshape(-1)
    parameter_retrain = np.asarray(parameter_retrain, dtype=float).reshape(-1)

    row: Dict[str, Any] = dict(base_info or {})
    row.update(
        {
            "method": method,
            "l1_constraint": np.nan if l1_constraint is None else float(l1_constraint),
            "param_l2_to_retrain": float(np.linalg.norm(parameter - parameter_retrain, 2)),
            "param_rel_l2_to_retrain": float(
                np.linalg.norm(parameter - parameter_retrain, 2)
                / (np.linalg.norm(parameter_retrain, 2) + 1e-12)
            ),
            "param_norm": float(np.linalg.norm(parameter, 2)),
            "param_retrain_norm": float(np.linalg.norm(parameter_retrain, 2)),
        }
    )

    if eps_info is not None:
        row.update(
            {
                "eps_l1": eps_info.get("eps_l1", np.nan),
                "eps_linf": eps_info.get("eps_linf", np.nan),
                "eps_min": eps_info.get("eps_min", np.nan),
                "eps_max": eps_info.get("eps_max", np.nan),
                "solver": eps_info.get("solver", None),
                "status": eps_info.get("status", None),
            }
        )

    metrics = evaluate_all(model, dataset_collection, cfg)
    row.update(flatten_metrics("metric", metrics))

    row = add_prediction_distances(
        row=row,
        model=model,
        model_retrain=model_retrain,
        dataset_collection=dataset_collection,
    )

    return row


def evaluate_retrain_distances(
    cfg: Any,
    model_original: torch.nn.Module,
    parameter_original: np.ndarray,
    model_complete: torch.nn.Module,
    parameter_complete: np.ndarray,
    model_retrain: torch.nn.Module,
    parameter_retrain: np.ndarray,
    dataset_collection: Dict[str, Any],
    repair_entries: Optional[List[Dict[str, Any]]] = None,
    base_info: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Evaluate original / complete / retrain / repair models against retrain.

    repair_entries format:
        [
            {
                "method": "repair_l1_0.15",
                "model": model_repair,
                "parameter": parameter_repair,
                "l1_constraint": 0.15,
                "eps_info": {...}
            },
            ...
        ]
    """
    rows = []

    rows.append(
        make_distance_row(
            method="original",
            model=model_original,
            parameter=parameter_original,
            model_retrain=model_retrain,
            parameter_retrain=parameter_retrain,
            dataset_collection=dataset_collection,
            cfg=cfg,
            base_info=base_info,
        )
    )

    rows.append(
        make_distance_row(
            method="complete",
            model=model_complete,
            parameter=parameter_complete,
            model_retrain=model_retrain,
            parameter_retrain=parameter_retrain,
            dataset_collection=dataset_collection,
            cfg=cfg,
            base_info=base_info,
        )
    )

    rows.append(
        make_distance_row(
            method="retrain",
            model=model_retrain,
            parameter=parameter_retrain,
            model_retrain=model_retrain,
            parameter_retrain=parameter_retrain,
            dataset_collection=dataset_collection,
            cfg=cfg,
            base_info=base_info,
        )
    )

    for entry in repair_entries or []:
        rows.append(
            make_distance_row(
                method=entry.get("method", "repair"),
                model=entry["model"],
                parameter=entry["parameter"],
                model_retrain=model_retrain,
                parameter_retrain=parameter_retrain,
                dataset_collection=dataset_collection,
                cfg=cfg,
                base_info=base_info,
                l1_constraint=entry.get("l1_constraint"),
                eps_info=entry.get("eps_info"),
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Result-dir loading helpers
# ---------------------------------------------------------------------
def load_repair_log(result_dir: str) -> Dict[str, Any]:
    path = os.path.join(result_dir, "repair_log.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"repair_log.npy not found: {path}")
    return np.load(path, allow_pickle=True).item()


def load_parameter(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.asarray(np.load(path), dtype=float).reshape(-1)


def load_repair_parameters_from_result_dir(
    result_dir: str,
    model_template: torch.nn.Module,
) -> List[Dict[str, Any]]:
    """
    Load repair parameters referenced in repair_log.npy and reconstruct models.

    Returns entries suitable for evaluate_retrain_distances().
    """
    log = load_repair_log(result_dir)

    entries = []
    for item in log.get("repair", []):
        l1 = float(item["l1_constraint"])

        parameter_path = item.get("parameter_path", None)
        if parameter_path is None:
            parameter_path = os.path.join(
                result_dir,
                f"parameter_repair_l1_{str(l1).replace('.', 'p')}.npy",
            )

        # If parameter_path was saved as an absolute/relative path from a previous cwd,
        # first try it as-is, then fallback to basename under result_dir.
        if not os.path.exists(parameter_path):
            parameter_path = os.path.join(result_dir, os.path.basename(parameter_path))

        parameter = load_parameter(parameter_path)
        model = reconstruct_model(model_template, parameter)

        eps_info = {
            "eps_l1": item.get("eps_l1", np.nan),
            "eps_linf": item.get("eps_linf", np.nan),
            "eps_min": item.get("eps_min", np.nan),
            "eps_max": item.get("eps_max", np.nan),
            "solver": item.get("solver", None),
            "status": item.get("status", None),
        }

        entries.append(
            {
                "method": f"repair_l1_{l1}",
                "model": model,
                "parameter": parameter,
                "l1_constraint": l1,
                "eps_info": eps_info,
            }
        )

    return entries


def compact_retrain_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "method",
        "index_mode",
        "index_criteria",
        "repair_criteria",
        "selection_policy",
        "unlearn_prop",
        "rho",
        "linf_constraint",
        "l1_constraint",
        "metric_mse_remain",
        "metric_mse_unlearn",
        "metric_mse_test",
        "metric_mape_remain",
        "metric_mape_unlearn",
        "metric_mape_test",
        "metric_cost_remain",
        "metric_cost_unlearn",
        "metric_cost_test",
        "param_l2_to_retrain",
        "param_rel_l2_to_retrain",
        "remain_pred_rmse_to_retrain",
        "unlearn_pred_rmse_to_retrain",
        "test_pred_rmse_to_retrain",
        "remain_pred_mape_to_retrain_percent",
        "unlearn_pred_mape_to_retrain_percent",
        "test_pred_mape_to_retrain_percent",
        "eps_l1",
        "eps_linf",
        "eps_min",
        "eps_max",
        "solver",
        "status",
    ]
    return df[[c for c in preferred if c in df.columns]].copy()
