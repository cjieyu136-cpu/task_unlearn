"""
utils/fed_affine_analytic.py

Exact analytic influence utilities for affine multi-output heads.

This is especially useful for topology-local-fusion NN variants where the
server head is still affine, but the fused feature dimension becomes much
larger and iterative CG IHVP can become numerically fragile.
"""

from typing import Any, Dict

import numpy as np

from utils.fed_local_head import (
    extract_affine_weight_bias,
    canonical_vec_to_model_order,
    model_order_vec_to_canonical,
)


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def augmented_feature_matrix(dataset) -> np.ndarray:
    X = _to_numpy(dataset.feature).astype(np.float64)
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    return np.concatenate([X, ones], axis=1)


def solve_affine_inverse_hvp_exact(
    model,
    dataset_remain,
    grad_model_order: np.ndarray,
    damping: float = 1e-8,
) -> Dict[str, Any]:
    """
    Exact -H^{-1} g for an affine multi-output head under mean-squared error.
    """
    feature_dim = int(dataset_remain.feature.shape[1])
    output_dim = int(dataset_remain.target.shape[1])

    grad_canonical = model_order_vec_to_canonical(
        model=model,
        feature_dim=feature_dim,
        output_dim=output_dim,
        model_order_vec=grad_model_order,
    )
    W_grad = grad_canonical[: output_dim * feature_dim].reshape(output_dim, feature_dim)
    b_grad = grad_canonical[output_dim * feature_dim:].reshape(output_dim)

    X_aug = augmented_feature_matrix(dataset_remain)
    N = int(X_aug.shape[0])
    H_base = (2.0 / float(N * output_dim)) * (X_aug.T @ X_aug)
    if float(damping) > 0:
        H_base = H_base + float(damping) * np.eye(H_base.shape[0], dtype=np.float64)

    W_M = np.zeros_like(W_grad, dtype=np.float64)
    b_M = np.zeros_like(b_grad, dtype=np.float64)
    solver = "solve"

    for j in range(output_dim):
        g_aug = np.concatenate([W_grad[j], [b_grad[j]]]).astype(np.float64)
        try:
            m_aug = -np.linalg.solve(H_base, g_aug)
        except np.linalg.LinAlgError:
            m_aug = -np.linalg.lstsq(H_base, g_aug, rcond=None)[0]
            solver = "lstsq"
        W_M[j] = m_aug[:-1]
        b_M[j] = m_aug[-1]

    M_canonical = np.concatenate([W_M.reshape(-1), b_M.reshape(-1)])
    M_model_order = canonical_vec_to_model_order(
        model=model,
        feature_dim=feature_dim,
        output_dim=output_dim,
        canonical_vec=M_canonical,
    )

    return {
        "M_model_order": M_model_order.astype(float),
        "M_canonical": M_canonical.astype(float),
        "info": {
            "solver": solver,
            "H_condition": float(np.linalg.cond(H_base)),
            "M_norm": float(np.linalg.norm(M_model_order)),
            "feature_dim": int(feature_dim),
            "output_dim": int(output_dim),
        },
    }


def compute_affine_sample_scores_exact(
    model,
    dataset_score,
    M_model_order: np.ndarray,
    normalize_by_n: bool = True,
) -> Dict[str, Any]:
    feature_dim = int(dataset_score.feature.shape[1])
    output_dim = int(dataset_score.target.shape[1])
    affine = extract_affine_weight_bias(model, feature_dim=feature_dim, output_dim=output_dim)

    parameter = np.concatenate([
        affine["W_out_in"].reshape(-1),
        affine["b"].reshape(-1),
    ]).astype(np.float64)
    M_canonical = model_order_vec_to_canonical(
        model=model,
        feature_dim=feature_dim,
        output_dim=output_dim,
        model_order_vec=M_model_order,
    )

    W = parameter[: output_dim * feature_dim].reshape(output_dim, feature_dim)
    b = parameter[output_dim * feature_dim:].reshape(output_dim)
    W_M = M_canonical[: output_dim * feature_dim].reshape(output_dim, feature_dim)
    b_M = M_canonical[output_dim * feature_dim:].reshape(output_dim)

    X = _to_numpy(dataset_score.feature).astype(np.float64)
    Y = _to_numpy(dataset_score.target).astype(np.float64)

    pred = X @ W.T + b.reshape(1, -1)
    err = pred - Y
    projected = X @ W_M.T + b_M.reshape(1, -1)

    score_by_bus = (2.0 / float(output_dim)) * err * projected
    scores = np.sum(score_by_bus, axis=1)
    if normalize_by_n and len(scores) > 0:
        scores = scores / float(len(scores))

    return {
        "scores": scores.astype(float),
        "info": {
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "num_scores": int(len(scores)),
            "normalize_by_n": bool(normalize_by_n),
        },
    }
