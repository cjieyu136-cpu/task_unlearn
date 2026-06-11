"""
utils/reweight_utils.py

Stage 2 utility module for TOP-FedTAMU+.

This module centralizes the repo-style PA-MU / TA-MU repair logic that was
validated in Stage 1.

It keeps the original repository's repair philosophy:

    DatasetWithWeight(dataset_remain, eps_remain)
    return_module(..., with_weight=True)
    ihvp = module.stest(...)
    parameter_repair = parameter_original - ihvp

and adds the Stage-1 fixes:

    1. index_criteria and repair_criteria are separated outside this module.
       This module only receives repair_criteria.

    2. The CVXPY L_inf constraint is implemented as explicit box constraints:
          1 - linf_constraint <= eps_i <= 1 + linf_constraint
       For linf_constraint=1, this is:
          0 <= eps_i <= 2

    3. Solver outputs are checked. Invalid eps is not silently accepted.

Main public functions
---------------------
compute_test_gradient_repo_style(...)
compute_inverse_hvp_vector(...)
compute_sample_scores(...)
solve_reweight_problem(...)
repo_style_complete_unlearning(...)
repo_style_repair_from_eps(...)
run_repo_style_repair_grid(...)

Typical usage
-------------
from utils.reweight_utils import (
    compute_test_gradient_repo_style,
    compute_inverse_hvp_vector,
    compute_sample_scores,
    solve_reweight_problem,
    repo_style_complete_unlearning,
    repo_style_repair_from_eps,
)

grad_test = compute_test_gradient_repo_style(...)
M_vec = compute_inverse_hvp_vector(...)
scores = compute_sample_scores(...)
eps, info = solve_reweight_problem(scores_remain, l1_constraint=0.15)
parameter_complete = repo_style_complete_unlearning(...)
parameter_repair = repo_style_repair_from_eps(...)
"""

import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import cvxpy as cp
import torch
from torch.utils.data import DataLoader

from utils import DatasetWithWeight, reconstruct_model
from func_operation import return_module


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def to_numpy(x: Any) -> np.ndarray:
    """Convert torch / list / numpy to a detached float numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def flatten_parameter_vector(parameter: Any) -> np.ndarray:
    """Return a flat float parameter vector."""
    return np.asarray(parameter, dtype=float).reshape(-1)


def get_model_parameter_vector(model: torch.nn.Module) -> np.ndarray:
    """
    Flatten model parameters in the repository-compatible order.

    Important:
        The original repo exposes utils.flatten_model(), and
        reconstruct_model() expects the same order. Do not rely on a custom
        flattening order here if utils.flatten_model is available.
    """
    try:
        from utils import flatten_model
        return np.asarray(flatten_model(model), dtype=float).reshape(-1)
    except Exception:
        vecs = []
        for p in model.parameters():
            vecs.append(to_numpy(p).reshape(-1))
        if not vecs:
            return np.array([], dtype=float)
        return np.concatenate(vecs).astype(float)


def _as_float_list(value: Any) -> List[float]:
    """Convert a scalar/list-like value to a Python list of floats."""
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in list(value)]


# ---------------------------------------------------------------------
# Module construction / gradients
# ---------------------------------------------------------------------
def build_repo_style_module(
    cfg: Any,
    model: torch.nn.Module,
    loader_train: DataLoader,
    loader_test: DataLoader,
    train_loss: str = "mse",
    test_loss: str = "mse",
    method: str = "cg",
    watch_progress: bool = False,
    with_weight: bool = False,
):
    """
    Thin wrapper around original return_module().

    This wrapper exists to keep Stage-2 code centralized and readable.
    """
    return return_module(
        cfg,
        loss_type_dict={"train": train_loss, "test": test_loss},
        loader_dict={"train": loader_train, "test": loader_test},
        model=model,
        method=method,
        watch_progress=watch_progress,
        with_weight=with_weight,
    )


def compute_test_gradient_repo_style(
    cfg: Any,
    model_test: torch.nn.Module,
    loader_train: DataLoader,
    loader_test: DataLoader,
    dataset_test: Any,
    repair_criteria: str,
    train_loss: str = "mse",
    method: str = "cg",
    watch_progress: bool = False,
) -> torch.Tensor:
    """
    Compute test gradient for the chosen repair criterion using the original
    repository's influence module.

    Important:
        - For repair_criteria=mse/mape, model_test is usually the affine model.
        - For repair_criteria=cost, model_test should be the SPO-wrapped model.

    This matches the fixedcrit Stage-1 implementation:
        module_test = return_module(... test=repair_criteria ...)
        grad_test = module_test.test_loss_grad(test_idxs=range(len(dataset_test)))
    """
    module_test = build_repo_style_module(
        cfg=cfg,
        model=model_test,
        loader_train=loader_train,
        loader_test=loader_test,
        train_loss=train_loss,
        test_loss=repair_criteria,
        method=method,
        watch_progress=watch_progress,
        with_weight=False,
    )

    # Important:
    # torch_influence.inverse_hvp expects `vec` to be a torch.Tensor because
    # modules.py uses vec.dtype as a torch dtype. Do NOT convert this gradient
    # to numpy before calling inverse_hvp.
    grad = module_test.test_loss_grad(test_idxs=range(len(dataset_test)))
    if isinstance(grad, torch.Tensor):
        return grad.detach().reshape(-1)
    return torch.as_tensor(grad, dtype=next(model_test.parameters()).dtype).reshape(-1)


def compute_inverse_hvp_vector(
    cfg: Any,
    model_train: torch.nn.Module,
    loader_train: DataLoader,
    vec: np.ndarray,
    train_loss: str = "mse",
    method: str = "cg",
    watch_progress: bool = False,
    sign: float = -1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute the original TAMU score direction:

        M = - H^{-1} grad_test

    By default, sign=-1, so:
        M_vec = - module.inverse_hvp(vec)

    Returns:
        M_vec, info
    """
    module_train = build_repo_style_module(
        cfg=cfg,
        model=model_train,
        loader_train=loader_train,
        loader_test=loader_train,
        train_loss=train_loss,
        test_loss=train_loss,
        method=method,
        watch_progress=watch_progress,
        with_weight=False,
    )

    # The original torch_influence implementation expects vec.dtype to be a
    # torch.dtype. If vec is numpy, inverse_hvp fails with:
    #   TypeError: tensor(): argument 'dtype' must be torch.dtype, not numpy.dtype
    # Therefore we explicitly keep/convert vec as a torch tensor before calling
    # inverse_hvp. This mirrors the original repository flow, where
    # test_loss_grad() returns a torch tensor and is passed directly to
    # inverse_hvp().
    if isinstance(vec, torch.Tensor):
        vec_torch = vec.detach().reshape(-1)
    else:
        first_param = next(model_train.parameters())
        vec_torch = torch.as_tensor(
            vec,
            dtype=first_param.dtype,
            device=first_param.device,
        ).reshape(-1)

    start = time.time()
    ihvp = module_train.inverse_hvp(vec=vec_torch)
    elapsed = time.time() - start

    M_vec = float(sign) * to_numpy(ihvp).reshape(-1)

    info = {
        "inverse_hvp_time": float(elapsed),
        "M_norm": float(np.linalg.norm(M_vec)),
        "vec_norm": float(torch.linalg.norm(vec_torch).detach().cpu().item()),
        "sign": float(sign),
    }
    for attr_name, info_name in [
        ("last_cg_info", "cg_info"),
        ("last_cg_iterations", "cg_iterations"),
        ("last_cg_residual_norm", "cg_residual_norm"),
        ("last_cg_relative_residual", "cg_relative_residual"),
    ]:
        if hasattr(module_train, attr_name):
            info[info_name] = getattr(module_train, attr_name)
    return M_vec, info


def compute_sample_scores(
    cfg: Any,
    model_train: torch.nn.Module,
    loader_hessian: DataLoader,
    dataset_score: Any,
    M_vec: np.ndarray,
    loader_score: Optional[DataLoader] = None,
    train_loss: str = "mse",
    method: str = "cg",
    watch_progress: bool = False,
    normalize_by_n: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute per-sample influence/reweight scores:

        score_i = grad_i^T M

    Original eval_unchange.py constructs the module as:
        train_loader = loader_remain   # Hessian on D_remain
        test_loader  = loader_train    # per-sample gradients on full train set

    and then computes:
        grad_i = module_train.test_loss_grad(test_idxs=[i])
        score_i = grad_i @ M
        scores = scores / len(dataset_train)

    Therefore:
        loader_hessian: loader for D_remain, used by the influence module as train_loader.
        loader_score: loader for full score dataset, usually D_train_affine.
        dataset_score: dataset whose rows receive scores, usually D_train_affine.

    If loader_score is None, it falls back to loader_hessian, which is useful only
    when scoring the same dataset used for Hessian.
    """
    if loader_score is None:
        loader_score = loader_hessian

    module_train = build_repo_style_module(
        cfg=cfg,
        model=model_train,
        loader_train=loader_hessian,
        loader_test=loader_score,
        train_loss=train_loss,
        test_loss=train_loss,
        method=method,
        watch_progress=watch_progress,
        with_weight=False,
    )

    M_vec = np.asarray(M_vec, dtype=float).reshape(-1)
    scores = []

    start = time.time()
    for i in range(len(dataset_score)):
        # Original repo uses test_loss_grad here because the scored samples are
        # in module_train.test_loader, not necessarily in train_loader.
        grad_i = module_train.test_loss_grad(test_idxs=[i])
        grad_i = to_numpy(grad_i).reshape(-1)
        scores.append(float(np.dot(grad_i, M_vec)))

    scores = np.asarray(scores, dtype=float)
    if normalize_by_n and len(dataset_score) > 0:
        scores = scores / float(len(dataset_score))

    elapsed = time.time() - start

    info = {
        "score_time": float(elapsed),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "num_scores": int(len(scores)),
        "normalize_by_n": bool(normalize_by_n),
        "hessian_loader_size": int(len(loader_hessian.dataset)),
        "score_dataset_size": int(len(dataset_score)),
    }
    return scores, info


# ---------------------------------------------------------------------
# Reweight optimization
# ---------------------------------------------------------------------
def solve_reweight_problem(
    scores_remain: np.ndarray,
    l1_constraint: float,
    linf_constraint: float = 1.0,
    tol: float = 1e-5,
    solver_order: Optional[Iterable[str]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve PA-MU / TA-MU remain-sample reweighting problem.

        min_eps    eps^T scores_remain

        s.t.       ||eps - 1||_1 <= l1_constraint * N
                   1 - linf_constraint <= eps_i <= 1 + linf_constraint

    Explicit box constraints are used instead of cp.norm(..., "inf") for
    numerical stability.

    Returns:
        eps_value, info
    """
    scores_remain = np.asarray(scores_remain, dtype=float).reshape(-1)
    N = len(scores_remain)

    if N == 0:
        raise ValueError("scores_remain is empty.")

    eps = cp.Variable(N)

    lower_bound = 1.0 - float(linf_constraint)
    upper_bound = 1.0 + float(linf_constraint)

    objective = cp.Minimize(cp.sum(cp.multiply(eps, scores_remain)))
    constraints = [
        cp.norm(eps - 1.0, 1) <= float(l1_constraint) * N,
        eps >= lower_bound,
        eps <= upper_bound,
    ]

    problem = cp.Problem(objective, constraints)

    installed = cp.installed_solvers()

    if solver_order is None:
        preferred = ["GUROBI", "MOSEK", "CLARABEL", "OSQP", "SCS", "ECOS"]
        solver_order = [s for s in preferred if s in installed]
    else:
        solver_order = [s for s in solver_order if s in installed]

    if not solver_order:
        raise RuntimeError(f"No supported CVXPY solver found. Installed solvers: {installed}")

    last_error = None

    for solver_name in solver_order:
        try:
            start = time.time()
            problem.solve(solver=solver_name, verbose=verbose)
            elapsed = time.time() - start

            if eps.value is None:
                last_error = f"solver={solver_name} returned eps=None"
                continue

            eps_value = np.asarray(eps.value, dtype=float).reshape(-1)

            eps_l1 = float(np.linalg.norm(eps_value - 1.0, 1))
            eps_linf = float(np.linalg.norm(eps_value - 1.0, np.inf))
            eps_min = float(np.min(eps_value))
            eps_max = float(np.max(eps_value))

            violates_l1 = eps_l1 > float(l1_constraint) * N + tol
            violates_box = (eps_min < lower_bound - tol) or (eps_max > upper_bound + tol)
            violates_linf = eps_linf > float(linf_constraint) + tol

            if violates_l1 or violates_box or violates_linf:
                last_error = (
                    f"solver={solver_name} returned infeasible eps: "
                    f"l1={eps_l1}, linf={eps_linf}, min={eps_min}, max={eps_max}, "
                    f"status={problem.status}"
                )
                continue

            # Clip only tiny numerical noise after feasibility check.
            eps_value = np.clip(eps_value, lower_bound, upper_bound)

            eps_l1 = float(np.linalg.norm(eps_value - 1.0, 1))
            eps_linf = float(np.linalg.norm(eps_value - 1.0, np.inf))
            eps_min = float(np.min(eps_value))
            eps_max = float(np.max(eps_value))

            info = {
                "status": str(problem.status),
                "solver": str(solver_name),
                "solve_time": float(elapsed),
                "objective_value": float(problem.value) if problem.value is not None else np.nan,
                "l1_constraint": float(l1_constraint),
                "linf_constraint": float(linf_constraint),
                "eps_l1": eps_l1,
                "eps_linf": eps_linf,
                "eps_min": eps_min,
                "eps_max": eps_max,
                "num_weights": int(N),
            }
            return eps_value, info

        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"CVXPY failed or returned invalid eps. Last error: {last_error}")


# ---------------------------------------------------------------------
# Repo-style complete / repair updates
# ---------------------------------------------------------------------
def repo_style_complete_unlearning(
    cfg: Any,
    model_original: torch.nn.Module,
    loader_remain: DataLoader,
    dataset_remain: Any,
    parameter_original: np.ndarray,
    train_loss: str = "mse",
    method: str = "cg",
    watch_progress: bool = False,
) -> Tuple[np.ndarray, torch.nn.Module, Dict[str, Any]]:
    """
    Original repo-style complete unlearning:

        module_unlearn = return_module(... loader_train=loader_remain ...)
        ihvp = module_unlearn.stest(test_idxs=range(len(dataset_remain)))
        parameter_complete = parameter_original - ihvp

    Returns:
        parameter_complete, model_complete, info
    """
    parameter_original = flatten_parameter_vector(parameter_original)

    module_unlearn = build_repo_style_module(
        cfg=cfg,
        model=model_original,
        loader_train=loader_remain,
        loader_test=loader_remain,
        train_loss=train_loss,
        test_loss=train_loss,
        method=method,
        watch_progress=watch_progress,
        with_weight=False,
    )

    start = time.time()
    ihvp = module_unlearn.stest(test_idxs=range(len(dataset_remain)))
    elapsed = time.time() - start

    ihvp = to_numpy(ihvp).reshape(-1)
    parameter_complete = parameter_original - ihvp
    model_complete = reconstruct_model(model_original, parameter_complete)

    info = {
        "complete_stest_time": float(elapsed),
        "ihvp_norm": float(np.linalg.norm(ihvp)),
        "parameter_complete_norm": float(np.linalg.norm(parameter_complete)),
    }
    for attr_name, info_name in [
        ("last_cg_info", "cg_info"),
        ("last_cg_iterations", "cg_iterations"),
        ("last_cg_residual_norm", "cg_residual_norm"),
        ("last_cg_relative_residual", "cg_relative_residual"),
    ]:
        if hasattr(module_unlearn, attr_name):
            info[info_name] = getattr(module_unlearn, attr_name)

    return parameter_complete, model_complete, info


def repo_style_repair_from_eps(
    cfg: Any,
    model_original: torch.nn.Module,
    dataset_remain: Any,
    eps_remain: np.ndarray,
    parameter_original: np.ndarray,
    batch_size: int,
    train_loss: str = "mse",
    method: str = "cg",
    watch_progress: bool = False,
) -> Tuple[np.ndarray, torch.nn.Module, Dict[str, Any]]:
    """
    Original repo-style weighted remain repair:

        dataset_remain_with_weight = DatasetWithWeight(dataset_remain, eps_remain)
        module = return_module(... with_weight=True)
        ihvp_weighted = module.stest(test_idxs=range(len(dataset_remain)))
        parameter_repair = parameter_original - ihvp_weighted

    Returns:
        parameter_repair, model_repair, info
    """
    parameter_original = flatten_parameter_vector(parameter_original)
    eps_remain = np.asarray(eps_remain, dtype=float).reshape(-1)

    if len(eps_remain) != len(dataset_remain):
        raise ValueError(
            f"eps_remain length mismatch: len(eps)={len(eps_remain)}, "
            f"len(dataset_remain)={len(dataset_remain)}"
        )

    dataset_weighted = DatasetWithWeight(dataset_remain, eps_remain)
    loader_weighted = DataLoader(
        dataset_weighted,
        batch_size=int(batch_size),
        shuffle=False,
    )

    module_weighted = build_repo_style_module(
        cfg=cfg,
        model=model_original,
        loader_train=loader_weighted,
        loader_test=loader_weighted,
        train_loss=train_loss,
        test_loss=train_loss,
        method=method,
        watch_progress=watch_progress,
        with_weight=True,
    )

    start = time.time()
    ihvp_weighted = module_weighted.stest(test_idxs=range(len(dataset_remain)))
    elapsed = time.time() - start

    ihvp_weighted = to_numpy(ihvp_weighted).reshape(-1)
    parameter_repair = parameter_original - ihvp_weighted
    model_repair = reconstruct_model(model_original, parameter_repair)

    info = {
        "weighted_stest_time": float(elapsed),
        "weighted_ihvp_norm": float(np.linalg.norm(ihvp_weighted)),
        "parameter_repair_norm": float(np.linalg.norm(parameter_repair)),
        "eps_l1": float(np.linalg.norm(eps_remain - 1.0, 1)),
        "eps_linf": float(np.linalg.norm(eps_remain - 1.0, np.inf)),
        "eps_min": float(np.min(eps_remain)),
        "eps_max": float(np.max(eps_remain)),
    }
    for attr_name, info_name in [
        ("last_cg_info", "cg_info"),
        ("last_cg_iterations", "cg_iterations"),
        ("last_cg_residual_norm", "cg_residual_norm"),
        ("last_cg_relative_residual", "cg_relative_residual"),
    ]:
        if hasattr(module_weighted, attr_name):
            info[info_name] = getattr(module_weighted, attr_name)

    return parameter_repair, model_repair, info


def run_repo_style_repair_grid(
    cfg: Any,
    model_original: torch.nn.Module,
    dataset_remain: Any,
    scores_remain: np.ndarray,
    parameter_original: np.ndarray,
    l1_constraints: Iterable[float],
    linf_constraint: float = 1.0,
    batch_size: Optional[int] = None,
    train_loss: str = "mse",
    method: str = "cg",
    watch_progress: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convenience function:
        for each l1_constraint:
            solve eps
            compute repo-style repair model / parameter

    Returns:
        list of dicts with:
            l1_constraint
            eps_remain
            parameter_repair
            model_repair
            reweight_info
            repair_info
    """
    if batch_size is None:
        batch_size = int(getattr(getattr(cfg, "data", object()), "batch_size_eval", 128))

    results: List[Dict[str, Any]] = []

    for l1_constraint in l1_constraints:
        eps_remain, reweight_info = solve_reweight_problem(
            scores_remain=scores_remain,
            l1_constraint=float(l1_constraint),
            linf_constraint=float(linf_constraint),
        )

        parameter_repair, model_repair, repair_info = repo_style_repair_from_eps(
            cfg=cfg,
            model_original=model_original,
            dataset_remain=dataset_remain,
            eps_remain=eps_remain,
            parameter_original=parameter_original,
            batch_size=int(batch_size),
            train_loss=train_loss,
            method=method,
            watch_progress=watch_progress,
        )

        results.append(
            {
                "l1_constraint": float(l1_constraint),
                "eps_remain": eps_remain,
                "parameter_repair": parameter_repair,
                "model_repair": model_repair,
                "reweight_info": reweight_info,
                "repair_info": repair_info,
            }
        )

    return results
