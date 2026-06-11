"""
utils/fed_block_influence.py

Stage 3F utility: client-local Hessian / block influence audit.

This module is intentionally an AUDIT utility, not a full repair implementation.

It tests the stricter federated approximation:

    M_k = -H_k^{-1} g_k
    score_i = Σ_k ∇_{θ_k} l_{i,k}^T M_k

where each bus client owns local output-head parameters:

    θ_k = (W_k, b_k)

Important boundary
------------------
This changes the influence approximation from the original repo-style global
Hessian:

    M = -H^{-1} g

to a block-diagonal approximation:

    H ≈ blockdiag(H_1, ..., H_K)

Therefore this module must be used as an audit first. If M_block or score_block
does not match the repo-style global baseline, that may be a mathematical
approximation effect, not necessarily a code bug.

Scaling convention
------------------
The repo-style train loss is MSE averaged over samples and output dimensions:

    L = mean_{i,b} (ŷ_{i,b} - y_{i,b})²

For local bus group B_k:

    L_k = (1 / (N * output_dim)) Σ_i Σ_{b∈B_k} (ŷ_{i,b} - y_{i,b})²

Thus for one sample i:

    l_{i,k} = (1 / output_dim) Σ_{b∈B_k} (ŷ_{i,b} - y_{i,b})²

This is equivalent to weighting the local mean by |B_k| / output_dim.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.fed_vjp_utils import (
    cloud_cost_gradient_scaled_for_dataset,
    model_output_scaled,
    alignment_metrics,
    get_dataset_mean_std,
)
from utils.fed_bus_client import parse_bus_groups, bus_groups_to_string
from utils.fed_local_head import (
    extract_affine_weight_bias,
    canonical_vec_to_model_order,
)


# ---------------------------------------------------------------------
# Parameter-order conversion
# ---------------------------------------------------------------------
def model_order_vec_to_canonical(
    model: torch.nn.Module,
    feature_dim: int,
    output_dim: int,
    vec_model_order: np.ndarray,
) -> np.ndarray:
    """
    Convert a vector in repo model.parameters() flatten order into canonical
    local-head order:

        [W_out_in.flatten(), b.flatten()]

    where W_out_in shape is [output_dim, feature_dim].
    """
    info = extract_affine_weight_bias(
        model,
        feature_dim=feature_dim,
        output_dim=output_dim,
    )

    vec = np.asarray(vec_model_order, dtype=float).reshape(-1)

    params = list(model.parameters())
    offsets = []
    start = 0
    for p in params:
        end = start + int(p.numel())
        offsets.append((start, end, tuple(p.shape)))
        start = end

    if vec.size != start:
        raise ValueError(f"vec has {vec.size} elements, model has {start}")

    w_start, w_end, w_shape = offsets[info["weight_index"]]
    b_start, b_end, b_shape = offsets[info["bias_index"]]

    W_raw = vec[w_start:w_end].reshape(w_shape)
    b = vec[b_start:b_end].reshape(b_shape)

    if info["orientation"] == "out_in":
        W_out_in = W_raw
    elif info["orientation"] == "in_out":
        W_out_in = W_raw.T
    else:
        raise ValueError(f"Unknown orientation: {info['orientation']}")

    return np.concatenate([W_out_in.reshape(-1), b.reshape(-1)]).astype(float)


def canonical_vec_to_weight_bias(
    canonical_vec: np.ndarray,
    feature_dim: int,
    output_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    canonical_vec = np.asarray(canonical_vec, dtype=float).reshape(-1)
    expected = output_dim * feature_dim + output_dim
    if canonical_vec.size != expected:
        raise ValueError(f"canonical_vec size {canonical_vec.size} != expected {expected}")

    W = canonical_vec[: output_dim * feature_dim].reshape(output_dim, feature_dim)
    b = canonical_vec[output_dim * feature_dim :]
    return W, b


# ---------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------
def dataset_arrays(dataset: Any) -> Tuple[np.ndarray, np.ndarray]:
    X = dataset.feature.detach().cpu().numpy() if isinstance(dataset.feature, torch.Tensor) else np.asarray(dataset.feature)
    Y = dataset.target.detach().cpu().numpy() if isinstance(dataset.target, torch.Tensor) else np.asarray(dataset.target)
    return X.astype(np.float64), Y.astype(np.float64)


def collect_model_predictions(
    model: torch.nn.Module,
    dataset: Any,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> np.ndarray:
    dev = torch.device(device or "cpu")
    model = model.to(dev)
    model.eval()

    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    preds = []
    with torch.no_grad():
        for feature, _target in loader:
            feature = feature.to(dev)
            pred = model_output_scaled(model, feature)
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float64)


# ---------------------------------------------------------------------
# Local cost gradient g_k
# ---------------------------------------------------------------------
def cost_gradient_canonical_by_bus(
    cfg: Any,
    model: torch.nn.Module,
    dataset_test: Any,
    num_clients: int = 4,
    bus_groups: Optional[str] = None,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute local-head cost gradient in canonical [W_out_in, b] order.

    CloudServer computes full gy. Each bus client would receive gy[:, B_k].
    For the affine head, each bus gradient is:

        d scalar / d W_b = Σ_i gy_{i,b} x_i
        d scalar / d b_b = Σ_i gy_{i,b}
    """
    X, _Y = dataset_arrays(dataset_test)
    feature_dim = int(X.shape[1])

    cloud = cloud_cost_gradient_scaled_for_dataset(
        cfg=cfg,
        model=model,
        dataset=dataset_test,
        batch_size=batch_size,
        device=device,
    )
    gy = np.asarray(cloud["gy_scaled"], dtype=np.float64)
    output_dim = int(gy.shape[1])

    groups = parse_bus_groups(bus_groups=bus_groups, num_bus=output_dim, num_clients=num_clients)

    W_grad = np.zeros((output_dim, feature_dim), dtype=np.float64)
    b_grad = np.zeros((output_dim,), dtype=np.float64)

    per_client = []
    for cid, buses in enumerate(groups):
        buses = np.asarray(buses, dtype=int)
        gy_k = gy[:, buses]

        Wg_k = gy_k.T @ X
        bg_k = np.sum(gy_k, axis=0)

        W_grad[buses, :] = Wg_k
        b_grad[buses] = bg_k

        per_client.append({
            "client_id": int(cid),
            "bus_indices": ",".join(str(int(x)) for x in buses),
            "num_buses": int(len(buses)),
            "g_k_elements": int(len(buses) * (feature_dim + 1)),
            "g_k_norm": float(np.sqrt(np.sum(Wg_k ** 2) + np.sum(bg_k ** 2))),
            "mean_abs_gy": float(np.mean(np.abs(gy_k))),
            "max_abs_gy": float(np.max(np.abs(gy_k))),
        })

    canonical = np.concatenate([W_grad.reshape(-1), b_grad.reshape(-1)]).astype(float)

    return {
        "grad_canonical": canonical,
        "W_grad_out_in": W_grad,
        "b_grad": b_grad,
        "gy_scaled": gy,
        "bus_groups": groups,
        "bus_groups_string": bus_groups_to_string(groups),
        "cloud_info": cloud["info"],
        "per_client": per_client,
        "info": {
            "feature_dim": feature_dim,
            "output_dim": output_dim,
            "num_clients": int(len(groups)),
            "num_samples": int(X.shape[0]),
            "bus_groups": bus_groups_to_string(groups),
            "grad_norm": float(np.linalg.norm(canonical)),
        },
    }


def criterion_gradient_canonical_by_bus(
    cfg: Any,
    model: torch.nn.Module,
    dataset_test: Any,
    repair_criteria: str,
    num_clients: int = 4,
    bus_groups: Optional[str] = None,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute local-head test gradient in canonical [W_out_in, b] order for
    mse / mape / cost repair criteria.
    """
    repair_criteria = str(repair_criteria).lower()
    if repair_criteria == "cost":
        return cost_gradient_canonical_by_bus(
            cfg=cfg,
            model=model,
            dataset_test=dataset_test,
            num_clients=num_clients,
            bus_groups=bus_groups,
            batch_size=batch_size,
            device=device,
        )

    if repair_criteria not in ["mse", "mape"]:
        raise ValueError(f"Unsupported repair_criteria for block_Hk: {repair_criteria}")

    X, Y = dataset_arrays(dataset_test)
    feature_dim = int(X.shape[1])
    output_dim = int(Y.shape[1])
    groups = parse_bus_groups(bus_groups=bus_groups, num_bus=output_dim, num_clients=num_clients)

    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.eval()

    loader = DataLoader(dataset_test, batch_size=int(batch_size), shuffle=False)
    preds_scaled = []
    with torch.no_grad():
        for feature, _target in loader:
            feature = feature.to(dev)
            pred = model_output_scaled(model, feature)
            preds_scaled.append(pred.detach().cpu())
    pred_scaled = torch.cat(preds_scaled, dim=0)

    target_scaled = torch.as_tensor(Y, dtype=pred_scaled.dtype)
    mean, std = get_dataset_mean_std(dataset_test, dtype=pred_scaled.dtype, device=pred_scaled.device)
    pred_unscaled = pred_scaled * std + mean
    target_unscaled = target_scaled * std.cpu() + mean.cpu()

    n = max(int(pred_scaled.shape[0]), 1)
    d = max(int(pred_scaled.shape[1]), 1)
    denom = float(n * d)

    # Match func_operation.return_objective():
    # - MSE_MSE.test_loss = mean((pred_scaled - target_scaled)^2)
    #   => d loss / d pred_scaled = 2 * (pred_scaled - target_scaled) / (n * d)
    # - MSE_MAPE.test_loss = mean(abs(pred_unscaled - target_unscaled) / target_unscaled)
    #   => d loss / d pred_scaled = sign(diff) * std / target_unscaled / (n * d)
    # There is no extra std^2 factor for MSE and no *100 factor for MAPE.
    if repair_criteria == "mse":
        gy = (2.0 * (pred_scaled - target_scaled)) / denom
    else:
        safe_target = torch.clamp(target_unscaled, min=1e-8)
        gy = torch.sign(pred_unscaled - target_unscaled) * ((std.cpu() / safe_target) / denom)

    gy = gy.numpy().astype(np.float64)

    W_grad = np.zeros((output_dim, feature_dim), dtype=np.float64)
    b_grad = np.zeros((output_dim,), dtype=np.float64)
    per_client = []
    for cid, buses in enumerate(groups):
        buses = np.asarray(buses, dtype=int)
        gy_k = gy[:, buses]
        Wg_k = gy_k.T @ X
        bg_k = np.sum(gy_k, axis=0)
        W_grad[buses, :] = Wg_k
        b_grad[buses] = bg_k
        per_client.append({
            "client_id": int(cid),
            "bus_indices": ",".join(str(int(x)) for x in buses),
            "num_buses": int(len(buses)),
            "g_k_elements": int(len(buses) * (feature_dim + 1)),
            "g_k_norm": float(np.sqrt(np.sum(Wg_k ** 2) + np.sum(bg_k ** 2))),
            "mean_abs_gy": float(np.mean(np.abs(gy_k))),
            "max_abs_gy": float(np.max(np.abs(gy_k))),
        })

    canonical = np.concatenate([W_grad.reshape(-1), b_grad.reshape(-1)]).astype(float)
    return {
        "grad_canonical": canonical,
        "W_grad_out_in": W_grad,
        "b_grad": b_grad,
        "gy_scaled": gy,
        "bus_groups": groups,
        "bus_groups_string": bus_groups_to_string(groups),
        "cloud_info": {"repair_criteria": repair_criteria},
        "per_client": per_client,
        "info": {
            "feature_dim": feature_dim,
            "output_dim": output_dim,
            "num_clients": int(len(groups)),
            "num_samples": int(X.shape[0]),
            "bus_groups": bus_groups_to_string(groups),
            "grad_norm": float(np.linalg.norm(canonical)),
            "repair_criteria": repair_criteria,
        },
    }


# ---------------------------------------------------------------------
# Block Hessian and IHVP
# ---------------------------------------------------------------------
def local_mse_hessian_augmented(
    dataset_remain: Any,
    output_dim: int,
    damping: float = 1e-8,
) -> np.ndarray:
    """
    Hessian for one output bus local affine parameters [w_b, b_b] under:

        L = (1 / (N * output_dim)) Σ_i error_{i,b}²

    For parameters [w, b], Hessian is:

        H = 2 / (N * output_dim) * X_aug^T X_aug
    """
    X, _Y = dataset_arrays(dataset_remain)
    N = int(X.shape[0])
    X_aug = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)

    H = (2.0 / float(N * output_dim)) * (X_aug.T @ X_aug)

    if damping is not None and float(damping) > 0:
        H = H + float(damping) * np.eye(H.shape[0], dtype=np.float64)

    return H.astype(np.float64)


def solve_block_ihvp(
    dataset_remain: Any,
    grad_canonical: np.ndarray,
    bus_groups: Sequence[np.ndarray],
    feature_dim: int,
    output_dim: int,
    damping: float = 1e-8,
) -> Dict[str, Any]:
    """
    Compute block-local:

        M_k = -H_k^{-1} g_k

    Since local MSE Hessian is identical for each bus under affine output heads,
    we solve one augmented Hessian per output bus.
    """
    Wg, bg = canonical_vec_to_weight_bias(
        grad_canonical,
        feature_dim=feature_dim,
        output_dim=output_dim,
    )

    H = local_mse_hessian_augmented(
        dataset_remain=dataset_remain,
        output_dim=output_dim,
        damping=damping,
    )

    W_M = np.zeros_like(Wg, dtype=np.float64)
    b_M = np.zeros_like(bg, dtype=np.float64)

    per_client = []

    for cid, buses in enumerate(bus_groups):
        buses = np.asarray(buses, dtype=int)
        block_norm_sq = 0.0

        for b in buses:
            g_aug = np.concatenate([Wg[b, :], np.asarray([bg[b]], dtype=np.float64)])

            try:
                m_aug = -np.linalg.solve(H, g_aug)
                solver = "solve"
            except np.linalg.LinAlgError:
                m_aug = -np.linalg.lstsq(H, g_aug, rcond=None)[0]
                solver = "lstsq"

            W_M[b, :] = m_aug[:-1]
            b_M[b] = m_aug[-1]
            block_norm_sq += float(np.sum(m_aug ** 2))

        per_client.append({
            "client_id": int(cid),
            "bus_indices": ",".join(str(int(x)) for x in buses),
            "num_buses": int(len(buses)),
            "M_k_elements": int(len(buses) * (feature_dim + 1)),
            "M_k_norm": float(np.sqrt(block_norm_sq)),
            "solver": solver,
        })

    M_canonical = np.concatenate([W_M.reshape(-1), b_M.reshape(-1)]).astype(float)

    return {
        "M_canonical": M_canonical,
        "W_M_out_in": W_M,
        "b_M": b_M,
        "H_local_augmented": H,
        "per_client": per_client,
        "info": {
            "feature_dim": feature_dim,
            "output_dim": output_dim,
            "damping": float(damping),
            "H_shape": tuple(H.shape),
            "H_condition": float(np.linalg.cond(H)),
            "M_norm": float(np.linalg.norm(M_canonical)),
        },
    }


# ---------------------------------------------------------------------
# Block sample scores
# ---------------------------------------------------------------------
def block_sample_scores_mse(
    model: torch.nn.Module,
    dataset_score: Any,
    M_canonical: np.ndarray,
    bus_groups: Sequence[np.ndarray],
    batch_size: int = 128,
    device: Optional[str] = None,
    normalize_by_n: bool = True,
) -> Dict[str, Any]:
    """
    Compute block-local sample scores:

        score_i = Σ_k ∇_{θ_k} l_{i,k}^T M_k

    with

        l_{i,k} = (1 / output_dim) Σ_{b∈B_k} (ŷ_{i,b} - y_{i,b})²

    For affine head, per sample and bus:

        ∂l_i/∂w_b = 2/output_dim * error_{i,b} * x_i
        ∂l_i/∂b_b = 2/output_dim * error_{i,b}

    If normalize_by_n=True, divide final scores by global len(dataset_score),
    matching repo-style compute_sample_scores(... normalize_by_n=True).
    """
    X, Y = dataset_arrays(dataset_score)
    N = int(X.shape[0])
    feature_dim = int(X.shape[1])
    output_dim = int(Y.shape[1])

    W_M, b_M = canonical_vec_to_weight_bias(
        M_canonical,
        feature_dim=feature_dim,
        output_dim=output_dim,
    )

    pred = collect_model_predictions(
        model=model,
        dataset=dataset_score,
        batch_size=batch_size,
        device=device,
    )

    err = pred - Y

    # Per bus contribution:
    # 2/output_dim * err_{i,b} * (x_i @ M_w_b + M_b_b)
    projected = X @ W_M.T + b_M.reshape(1, -1)
    score_by_bus = (2.0 / float(output_dim)) * err * projected

    scores = np.sum(score_by_bus, axis=1)

    if normalize_by_n:
        scores = scores / float(N)

    per_client = []
    for cid, buses in enumerate(bus_groups):
        buses = np.asarray(buses, dtype=int)
        local_score = np.sum(score_by_bus[:, buses], axis=1)
        if normalize_by_n:
            local_score = local_score / float(N)

        per_client.append({
            "client_id": int(cid),
            "bus_indices": ",".join(str(int(x)) for x in buses),
            "num_buses": int(len(buses)),
            "score_min": float(np.min(local_score)),
            "score_max": float(np.max(local_score)),
            "score_mean": float(np.mean(local_score)),
            "score_std": float(np.std(local_score)),
        })

    return {
        "scores": scores.astype(float),
        "per_client": per_client,
        "info": {
            "num_samples": int(N),
            "feature_dim": feature_dim,
            "output_dim": output_dim,
            "normalize_by_n": bool(normalize_by_n),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
        },
    }


# ---------------------------------------------------------------------
# End-to-end block audit helper
# ---------------------------------------------------------------------
def block_influence_audit(
    cfg: Any,
    model: torch.nn.Module,
    dataset_test: Any,
    dataset_remain: Any,
    dataset_score: Any,
    M_global_model_order: np.ndarray,
    scores_global: np.ndarray,
    num_clients: int = 4,
    bus_groups: Optional[str] = None,
    batch_size: int = 128,
    block_damping: float = 1e-8,
    device: Optional[str] = None,
    repair_criteria: str = "cost",
) -> Dict[str, Any]:
    """
    Compute block-local cost gradient, block IHVP, and block sample scores,
    then compare them to repo-style global M and scores.
    """
    feature_dim = int(dataset_score.feature.shape[1])
    output_dim = int(dataset_score.target.shape[1])

    g_result = criterion_gradient_canonical_by_bus(
        cfg=cfg,
        model=model,
        dataset_test=dataset_test,
        repair_criteria=repair_criteria,
        num_clients=num_clients,
        bus_groups=bus_groups,
        batch_size=batch_size,
        device=device,
    )

    groups = g_result["bus_groups"]

    M_block = solve_block_ihvp(
        dataset_remain=dataset_remain,
        grad_canonical=g_result["grad_canonical"],
        bus_groups=groups,
        feature_dim=feature_dim,
        output_dim=output_dim,
        damping=block_damping,
    )

    scores_block = block_sample_scores_mse(
        model=model,
        dataset_score=dataset_score,
        M_canonical=M_block["M_canonical"],
        bus_groups=groups,
        batch_size=batch_size,
        device=device,
        normalize_by_n=True,
    )

    M_global_canonical = model_order_vec_to_canonical(
        model=model,
        feature_dim=feature_dim,
        output_dim=output_dim,
        vec_model_order=M_global_model_order,
    )

    M_alignment = alignment_metrics(M_global_canonical, M_block["M_canonical"])
    score_alignment = alignment_metrics(scores_global, scores_block["scores"])

    M_block_model_order = canonical_vec_to_model_order(
        model=model,
        feature_dim=feature_dim,
        output_dim=output_dim,
        canonical_vec=M_block["M_canonical"],
    )

    return {
        "grad_result": g_result,
        "M_block": M_block,
        "scores_block": scores_block,
        "M_global_canonical": M_global_canonical,
        "M_block_model_order": M_block_model_order,
        "M_alignment": M_alignment,
        "score_alignment": score_alignment,
        "info": {
            "feature_dim": feature_dim,
            "output_dim": output_dim,
            "num_clients": int(len(groups)),
            "bus_groups": g_result["bus_groups_string"],
            "block_damping": float(block_damping),
        },
    }
