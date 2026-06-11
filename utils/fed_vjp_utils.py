"""
utils/fed_vjp_utils.py

Stage 3 utility module for Fed-TA-MU.

Core purpose
------------
Federated task-aware gradient decomposition:

    Cloud:
        g_y = d L_cost / d y_hat

    Client:
        grad_theta L_cost = J_f(theta)^T g_y

This module is NOT a new unlearning selector. It does not replace random /
helpful / harmful / event indices. It is an audit and implementation layer
for decomposing the original centralized TA-MU cost gradient into:

    1. cloud-side OPF/cost marginal signal, also called
       VJP-derived effective shadow price;
    2. client-side local VJP through the forecasting model.

Repository alignment
--------------------
The cost layer mirrors the original repository:

    utils.net.Stage_One_Layer
    utils.net.Stage_Two_Layer
    utils.optimization.Operator
    func_operation.return_module(... test_loss='cost')

The cost formula matches MSE_COST.test_loss in func_operation.py:

    loss_gen = pg^2 @ second_coeff + pg @ first_coeff
    loss_ls  = sum(ls^2) * load_shed_coeff_second + sum(ls) * load_shed_coeff
    loss_gs  = sum(gs^2) * gen_storage_coeff_second + sum(gs) * gen_storage_coeff

Important scaling convention
----------------------------
The topology-affine model outputs scaled load when the dataset is scaled.
The cloud computes gradient with respect to this scaled output:

    g_y_scaled = d L_cost / d y_hat_scaled

This is exactly what the client model sees, so the client VJP is:

    grad_theta = J_{y_hat_scaled}(theta)^T g_y_scaled

This avoids a mismatch between physical-load gradients and scaled model output.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.net import Stage_One_Layer, Stage_Two_Layer, SPO
from utils.optimization import Operator

from utils.reweight_utils import (
    compute_test_gradient_repo_style,
    get_model_parameter_vector,
)


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def model_output_scaled(model: torch.nn.Module, feature: torch.Tensor) -> torch.Tensor:
    """
    Return forecast output in the model's native scale.

    NN_CONV returns (feature, forecast); affine heads return forecast directly.
    """
    out = model(feature)
    if isinstance(out, (tuple, list)):
        return out[-1]
    return out


def get_dataset_mean_std(dataset: Any, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return target mean/std tensors for unscaling model output.
    """
    if getattr(dataset, "is_scale", False):
        mean = dataset.target_mean
        std = dataset.target_std
    else:
        mean = 0.0
        std = 1.0

    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=dtype, device=device)
    else:
        mean = mean.to(dtype=dtype, device=device)

    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=dtype, device=device)
    else:
        std = std.to(dtype=dtype, device=device)

    return mean, std


def flatten_grad_from_model(model: torch.nn.Module, grads: List[Optional[torch.Tensor]]) -> np.ndarray:
    """
    Flatten gradients in the same order as model.parameters().
    """
    chunks = []
    for p, g in zip(model.parameters(), grads):
        if g is None:
            chunks.append(torch.zeros_like(p).detach().reshape(-1).cpu().numpy())
        else:
            chunks.append(g.detach().reshape(-1).cpu().numpy())
    if not chunks:
        return np.array([], dtype=float)
    return np.concatenate(chunks).astype(float)


def alignment_metrics(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    """
    Compare two gradient vectors.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    if len(a) != len(b):
        raise ValueError(f"Gradient length mismatch: {len(a)} vs {len(b)}")

    diff = a - b
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    diff_norm = float(np.linalg.norm(diff))

    cosine = float(np.dot(a, b) / ((a_norm + eps) * (b_norm + eps)))
    rel_l2 = float(diff_norm / (a_norm + eps))
    norm_ratio = float(b_norm / (a_norm + eps))

    return {
        "centralized_norm": a_norm,
        "fed_vjp_norm": b_norm,
        "diff_norm": diff_norm,
        "relative_l2_error": rel_l2,
        "cosine_similarity": cosine,
        "norm_ratio_fed_over_centralized": norm_ratio,
        "max_abs_diff": float(np.max(np.abs(diff))) if len(diff) else 0.0,
        "mean_abs_diff": float(np.mean(np.abs(diff))) if len(diff) else 0.0,
    }


# ---------------------------------------------------------------------
# Cloud-side OPF/cost layer
# ---------------------------------------------------------------------
class CloudCostLayer(torch.nn.Module):
    """
    Cloud-side OPF/cost layer.

    It receives forecasted load and true load in physical units, runs the
    repository's stage-one and stage-two differentiable OPF layers, and returns
    the OPF-aware cost used by the original MSE_COST objective.
    """

    def __init__(self, cfg: Any, device: Optional[torch.device] = None):
        super().__init__()
        self.operator = Operator(case_config=cfg.case)
        self.device = device or torch.device("cpu")

        self.stage_one_layer = Stage_One_Layer(self.operator.prob_1).to(self.device)
        self.stage_two_layer = Stage_Two_Layer(self.operator.prob_2).to(self.device)

        self.second_coeff = torch.tensor(self.operator.second_coeff, dtype=torch.float32, device=self.device)
        self.first_coeff = torch.tensor(self.operator.first_coeff, dtype=torch.float32, device=self.device)

        self.load_shed_coeff_second = float(self.operator.load_shed_coeff_second)
        self.load_shed_coeff = float(self.operator.load_shed_coeff)
        self.gen_storage_coeff_second = float(self.operator.gen_storage_coeff_second)
        self.gen_storage_coeff = float(self.operator.gen_storage_coeff)

    def forward_cost_per_sample(
        self,
        forecast_load_physical: torch.Tensor,
        target_load_physical: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Return cost per sample and OPF primal variables.
        """
        forecast_load_physical = forecast_load_physical.to(self.device)
        target_load_physical = target_load_physical.to(self.device)

        pg = self.stage_one_layer(forecast_load_physical)
        ls, gs = self.stage_two_layer(pg, target_load_physical)

        second_coeff = self.second_coeff.to(dtype=pg.dtype, device=pg.device)
        first_coeff = self.first_coeff.to(dtype=pg.dtype, device=pg.device)

        loss_gen = torch.square(pg) @ second_coeff + pg @ first_coeff
        loss_ls = (
            torch.square(ls).sum(axis=1) * self.load_shed_coeff_second
            + ls.sum(axis=1) * self.load_shed_coeff
        )
        loss_gs = (
            torch.square(gs).sum(axis=1) * self.gen_storage_coeff_second
            + gs.sum(axis=1) * self.gen_storage_coeff
        )

        cost = loss_gen + loss_ls + loss_gs

        return cost, {"pg": pg, "ls": ls, "gs": gs}


def cloud_cost_gradient_scaled_for_dataset(
    cfg: Any,
    model: torch.nn.Module,
    dataset: Any,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Cloud computes g_y_scaled = d L_cost / d y_hat_scaled for every sample.

    The loss is averaged over the whole dataset, matching the original
    repository test_loss mean-reduction semantics.
    """
    dev = torch.device(device or "cpu")
    model = model.to(dev)
    model.eval()

    layer = CloudCostLayer(cfg=cfg, device=dev)

    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    n_total = len(dataset)

    all_forecast_scaled = []
    all_target_scaled = []
    all_gy_scaled = []
    all_cost = []

    for feature, target in loader:
        feature = feature.to(dev)
        target = target.to(dev)

        with torch.no_grad():
            pred_scaled_const = model_output_scaled(model, feature)

        pred_scaled = pred_scaled_const.detach().clone().requires_grad_(True)

        mean, std = get_dataset_mean_std(dataset, dtype=pred_scaled.dtype, device=dev)
        forecast_physical = pred_scaled * std + mean
        target_physical = target * std + mean

        cost_per_sample, opf = layer.forward_cost_per_sample(
            forecast_load_physical=forecast_physical,
            target_load_physical=target_physical,
        )

        # Mean over full dataset, not just current batch.
        loss = torch.sum(cost_per_sample) / float(n_total)

        if pred_scaled.grad is not None:
            pred_scaled.grad.zero_()

        loss.backward()

        all_forecast_scaled.append(pred_scaled_const.detach().cpu().numpy())
        all_target_scaled.append(target.detach().cpu().numpy())
        all_gy_scaled.append(pred_scaled.grad.detach().cpu().numpy())
        all_cost.append(cost_per_sample.detach().cpu().numpy())

    forecast_scaled = np.concatenate(all_forecast_scaled, axis=0).astype(float)
    target_scaled = np.concatenate(all_target_scaled, axis=0).astype(float)
    gy_scaled = np.concatenate(all_gy_scaled, axis=0).astype(float)
    cost = np.concatenate(all_cost, axis=0).astype(float)

    error_scaled = forecast_scaled - target_scaled
    first_order_proxy = np.sum(gy_scaled * error_scaled, axis=1)

    gy_abs = np.abs(gy_scaled)

    return {
        "forecast_scaled": forecast_scaled,
        "target_scaled": target_scaled,
        "gy_scaled": gy_scaled,
        "cost_per_sample": cost,
        "error_scaled": error_scaled,
        "first_order_cost_proxy": first_order_proxy,
        "sample_vjp_score_l2": np.linalg.norm(gy_scaled, axis=1),
        "sample_vjp_score_l1": np.sum(gy_abs, axis=1),
        "bus_vjp_score_l1": np.mean(gy_abs, axis=0),
        "bus_vjp_score_l2": np.sqrt(np.mean(gy_scaled ** 2, axis=0)),
        "info": {
            "num_samples": int(gy_scaled.shape[0]),
            "num_bus": int(gy_scaled.shape[1]),
            "mean_cost": float(np.mean(cost)),
            "mean_abs_gy": float(np.mean(gy_abs)),
            "max_abs_gy": float(np.max(gy_abs)),
            "loss_reduction": "sum(cost_per_sample) / num_samples",
        },
    }


def cloud_criterion_gradient_scaled_for_dataset(
    cfg: Any,
    model: torch.nn.Module,
    dataset: Any,
    repair_criteria: str,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute d L_test / d y_hat_scaled for cost / mse / mape.

    This is the runtime-side analogue of repo criterion selection:
        - cost: OPF-aware cloud gradient wrt scaled forecast
        - mse : mean squared error on scaled outputs
        - mape: mean absolute percentage error on unscaled outputs
    """
    repair_criteria = str(repair_criteria).lower()
    if repair_criteria == "cost":
        return cloud_cost_gradient_scaled_for_dataset(
            cfg=cfg,
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
        )
    if repair_criteria not in ["mse", "mape"]:
        raise ValueError(f"Unsupported repair_criteria: {repair_criteria}")

    dev = torch.device(device or "cpu")
    model = model.to(dev)
    model.eval()

    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    n_total = len(dataset)

    all_forecast_scaled = []
    all_target_scaled = []
    all_gy_scaled = []

    for feature, target in loader:
        feature = feature.to(dev)
        target = target.to(dev)

        with torch.no_grad():
            pred_scaled = model_output_scaled(model, feature)

        mean, std = get_dataset_mean_std(dataset, dtype=pred_scaled.dtype, device=dev)
        pred_unscaled = pred_scaled * std + mean
        target_unscaled = target * std + mean

        denom = float(max(int(n_total), 1) * max(int(pred_scaled.shape[1]), 1))
        if repair_criteria == "mse":
            gy_scaled = (2.0 * (pred_scaled - target)) / denom
        else:
            safe_target = torch.clamp(target_unscaled, min=1e-8)
            gy_scaled = torch.sign(pred_unscaled - target_unscaled) * ((std / safe_target) / denom)

        all_forecast_scaled.append(pred_scaled.detach().cpu().numpy())
        all_target_scaled.append(target.detach().cpu().numpy())
        all_gy_scaled.append(gy_scaled.detach().cpu().numpy())

    forecast_scaled = np.concatenate(all_forecast_scaled, axis=0).astype(float)
    target_scaled = np.concatenate(all_target_scaled, axis=0).astype(float)
    gy_scaled = np.concatenate(all_gy_scaled, axis=0).astype(float)
    gy_abs = np.abs(gy_scaled)

    return {
        "forecast_scaled": forecast_scaled,
        "target_scaled": target_scaled,
        "gy_scaled": gy_scaled,
        "error_scaled": forecast_scaled - target_scaled,
        "sample_vjp_score_l2": np.linalg.norm(gy_scaled, axis=1),
        "sample_vjp_score_l1": np.sum(gy_abs, axis=1),
        "bus_vjp_score_l1": np.mean(gy_abs, axis=0),
        "bus_vjp_score_l2": np.sqrt(np.mean(gy_scaled ** 2, axis=0)),
        "info": {
            "num_samples": int(gy_scaled.shape[0]),
            "num_bus": int(gy_scaled.shape[1]),
            "mean_abs_gy": float(np.mean(gy_abs)),
            "max_abs_gy": float(np.max(gy_abs)),
            "repair_criteria": repair_criteria,
        },
    }


def cloud_safety_proxy_for_dataset(
    cfg: Any,
    model: torch.nn.Module,
    dataset: Any,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Cloud computes sample-level safety proxy signals from the same OPF pipeline.

    The first minimal proxy is stage-two total load shedding per sample:

        risk_i = sum_j ls_stage2[i, j]
    """
    dev = torch.device(device or "cpu")
    model = model.to(dev)
    model.eval()

    layer = CloudCostLayer(cfg=cfg, device=dev)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)

    all_ls_total = []
    all_gs_total = []
    all_dispatch_gap_abs = []

    for feature, target in loader:
        feature = feature.to(dev)
        target = target.to(dev)

        with torch.no_grad():
            pred_scaled = model_output_scaled(model, feature)

        mean, std = get_dataset_mean_std(dataset, dtype=pred_scaled.dtype, device=dev)
        forecast_physical = pred_scaled * std + mean
        target_physical = target * std + mean

        _cost_per_sample, opf = layer.forward_cost_per_sample(
            forecast_load_physical=forecast_physical,
            target_load_physical=target_physical,
        )

        pg = opf["pg"]
        ls = opf["ls"]
        gs = opf["gs"]

        ls_total = torch.sum(ls, dim=1)
        gs_total = torch.sum(gs, dim=1)
        dispatch_gap_abs = torch.abs(torch.sum(pg, dim=1) - torch.sum(target_physical, dim=1))

        all_ls_total.append(ls_total.detach().cpu().numpy())
        all_gs_total.append(gs_total.detach().cpu().numpy())
        all_dispatch_gap_abs.append(dispatch_gap_abs.detach().cpu().numpy())

    ls_total = np.concatenate(all_ls_total, axis=0).astype(float)
    gs_total = np.concatenate(all_gs_total, axis=0).astype(float)
    dispatch_gap_abs = np.concatenate(all_dispatch_gap_abs, axis=0).astype(float)

    return {
        "stage2_ls_total": ls_total,
        "stage2_gs_total": gs_total,
        "dispatch_gap_abs": dispatch_gap_abs,
        "info": {
            "num_samples": int(ls_total.shape[0]),
            "stage2_ls_mean": float(np.mean(ls_total)),
            "stage2_ls_max": float(np.max(ls_total)),
            "stage2_gs_mean": float(np.mean(gs_total)),
            "dispatch_gap_abs_mean": float(np.mean(dispatch_gap_abs)),
        },
    }


# ---------------------------------------------------------------------
# Client-side VJP
# ---------------------------------------------------------------------
def client_vjp_from_gy(
    model: torch.nn.Module,
    dataset: Any,
    gy_scaled: np.ndarray,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Client computes J_f(theta)^T g_y.

    gy_scaled must be the cloud gradient wrt model output in the same scaled
    space as model(feature).
    """
    dev = torch.device(device or "cpu")
    model = model.to(dev)
    model.eval()

    gy_scaled = np.asarray(gy_scaled, dtype=np.float32)

    if gy_scaled.shape[0] != len(dataset):
        raise ValueError(f"gy_scaled length {gy_scaled.shape[0]} does not match dataset length {len(dataset)}")

    # clear grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)

    start = 0
    for feature, _target in loader:
        bsz = feature.shape[0]
        end = start + bsz

        feature = feature.to(dev)
        gy_batch = torch.tensor(gy_scaled[start:end], dtype=torch.float32, device=dev)

        pred = model_output_scaled(model, feature)

        # VJP: sum_i,k pred[i,k] * gy[i,k]
        scalar = torch.sum(pred * gy_batch)

        grads = torch.autograd.grad(
            scalar,
            list(model.parameters()),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        for p, g in zip(model.parameters(), grads):
            if g is None:
                continue
            if p.grad is None:
                p.grad = g.detach().clone()
            else:
                p.grad += g.detach()

        start = end

    grad_chunks = []
    for p in model.parameters():
        if p.grad is None:
            grad_chunks.append(torch.zeros_like(p).detach().reshape(-1).cpu().numpy())
        else:
            grad_chunks.append(p.grad.detach().reshape(-1).cpu().numpy())

    return np.concatenate(grad_chunks).astype(float)


def centralized_cost_gradient_repo_style(
    cfg: Any,
    model: torch.nn.Module,
    dataset: Any,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Compute centralized repo-style cost gradient using return_module/test_loss_grad.

    This is the reference for Fed-VJP alignment.
    """
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)

    operator = Operator(case_config=cfg.case)
    if getattr(dataset, "is_scale", False):
        mean = dataset.target_mean
        std = dataset.target_std
    else:
        mean = 0
        std = 1

    model_spo = SPO(
        trained_model=model,
        operator=operator,
        mean=mean,
        std=std,
    )
    model_spo.eval()

    grad = compute_test_gradient_repo_style(
        cfg=cfg,
        model_test=model_spo,
        loader_train=loader,
        loader_test=loader,
        dataset_test=dataset,
        repair_criteria="cost",
        train_loss="mse",
    )

    return to_numpy(grad).reshape(-1).astype(float)


def fed_vjp_alignment_audit(
    cfg: Any,
    model: torch.nn.Module,
    dataset: Any,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare centralized repo-style cost gradient with cloud-gradient + client-VJP.
    """
    cloud = cloud_cost_gradient_scaled_for_dataset(
        cfg=cfg,
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
    )

    grad_fed = client_vjp_from_gy(
        model=model,
        dataset=dataset,
        gy_scaled=cloud["gy_scaled"],
        batch_size=batch_size,
        device=device,
    )

    grad_central = centralized_cost_gradient_repo_style(
        cfg=cfg,
        model=model,
        dataset=dataset,
        batch_size=batch_size,
    )

    metrics = alignment_metrics(grad_central, grad_fed)

    return {
        "cloud": cloud,
        "grad_fed_vjp": grad_fed,
        "grad_centralized": grad_central,
        "alignment": metrics,
    }


# ---------------------------------------------------------------------
# Index-mode audit helpers
# ---------------------------------------------------------------------
def summarize_unlearn_vjp_scores(
    sample_scores: np.ndarray,
    unlearn_index: np.ndarray,
    remain_index: np.ndarray,
) -> Dict[str, float]:
    """
    Compare VJP task sensitivity on unlearn vs remain samples.
    """
    sample_scores = np.asarray(sample_scores, dtype=float).reshape(-1)
    unlearn_index = np.asarray(unlearn_index, dtype=int).reshape(-1)
    remain_index = np.asarray(remain_index, dtype=int).reshape(-1)

    unlearn_scores = sample_scores[unlearn_index]
    remain_scores = sample_scores[remain_index]

    return {
        "num_unlearn": int(len(unlearn_index)),
        "num_remain": int(len(remain_index)),
        "unlearn_vjp_mean": float(np.mean(unlearn_scores)),
        "remain_vjp_mean": float(np.mean(remain_scores)),
        "unlearn_vjp_median": float(np.median(unlearn_scores)),
        "remain_vjp_median": float(np.median(remain_scores)),
        "unlearn_vjp_std": float(np.std(unlearn_scores)),
        "remain_vjp_std": float(np.std(remain_scores)),
        "ratio_unlearn_vs_remain": float(np.mean(unlearn_scores) / (np.mean(remain_scores) + 1e-12)),
    }


def build_vjp_sample_table(
    cloud: Dict[str, Any],
    unlearn_index: np.ndarray,
) -> "pd.DataFrame":
    """
    Build per-sample VJP audit table.
    """
    import pandas as pd

    n = cloud["gy_scaled"].shape[0]
    is_unlearn = np.zeros(n, dtype=bool)
    is_unlearn[np.asarray(unlearn_index, dtype=int)] = True

    return pd.DataFrame(
        {
            "sample_index": np.arange(n, dtype=int),
            "is_unlearn": is_unlearn,
            "cost_per_sample": cloud["cost_per_sample"],
            "sample_vjp_score_l1": cloud["sample_vjp_score_l1"],
            "sample_vjp_score_l2": cloud["sample_vjp_score_l2"],
            "first_order_cost_proxy": cloud["first_order_cost_proxy"],
        }
    )


def build_vjp_bus_table(cloud: Dict[str, Any]) -> "pd.DataFrame":
    """
    Build bus-level VJP audit table.
    """
    import pandas as pd

    b = cloud["gy_scaled"].shape[1]
    return pd.DataFrame(
        {
            "bus_index": np.arange(b, dtype=int),
            "bus_vjp_score_l1": cloud["bus_vjp_score_l1"],
            "bus_vjp_score_l2": cloud["bus_vjp_score_l2"],
            "bus_gy_signed_mean": np.mean(cloud["gy_scaled"], axis=0),
            "bus_gy_signed_std": np.std(cloud["gy_scaled"], axis=0),
        }
    ).sort_values("bus_vjp_score_l1", ascending=False)
