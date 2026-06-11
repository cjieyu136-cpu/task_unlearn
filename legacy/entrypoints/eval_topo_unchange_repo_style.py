
"""
eval_topo_unchange_repo_style.py

Repository-style PA-MU / TA-MU repair alignment script.

Purpose
-------
This script follows the original repository eval_unchange.py repair path as closely as possible:

    1. Use DatasetWithWeight(dataset_remain, eps_remain)
    2. Build return_module(..., with_weight=True)
    3. Compute ihvp = module_unlearn.stest(test_idxs=range(len(dataset_remain)))
    4. parameter_repair = parameter_ori - ihvp
    5. reconstruct_model(model_ori, parameter_repair)

This script is for ALIGNMENT, not the final mask-aware method.

Important limitations
---------------------
- It supports sample-level / row-level unlearning:
    random, helpful, harmful, event, event_system
- If index_mode=event_mask, it is converted to row-level event times for alignment only.
  It does NOT implement true bus-time mask repair.
- It does not include topology Hessian in return_module. The original torch-influence module
  uses the ordinary training Hessian on the remain set.

Use it to answer:
    Can we reproduce the original TA-MU repair trend under
    unlearn_prop=0.2, index_mode=helpful, criteria=cost, rho=0?
"""

import os
import time
import numpy as np
import torch
import hydra
import cvxpy as cp

from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from utils import (
    return_dataset,
    NewDataset,
    DatasetWithWeight,
    evaluate,
    reconstruct_model,
    flatten_model,
)
from utils.optimization import Operator
from utils.net import SPO
from utils.topology import build_load_bus_adjacency, build_laplacian
from utils.topo_affine import return_topology_affine_model

from func_operation import (
    return_core_datasets,
    return_dataset_for_nn_affine,
    return_module,
)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _safe_int_seed(cfg):
    return int(cfg.data.random_seed)


def _format_float_for_path(x):
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def _as_float_list(value):
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return [float(value)]
    return [float(v) for v in list(value)]


def _get_l1_constraints(cfg):
    root_constraints = OmegaConf.select(cfg, "l1_constraints", default=None)
    if root_constraints is not None:
        parsed = _as_float_list(root_constraints)
        if parsed is not None:
            return parsed

    model_constraints = OmegaConf.select(cfg, "model.l1_constraints", default=None)
    if model_constraints is not None:
        parsed = _as_float_list(model_constraints)
        if parsed is not None:
            return parsed

    return [0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.0]


def split_dataset_by_index(dataset, unlearn_index):
    N = len(dataset)
    unlearn_index = np.asarray(unlearn_index).astype(int).reshape(-1)
    unlearn_index = unlearn_index[(unlearn_index >= 0) & (unlearn_index < N)]

    unlearn_set = set(unlearn_index.tolist())
    remain_index = np.array([i for i in range(N) if i not in unlearn_set], dtype=int)

    target_mean = getattr(dataset, "target_mean", 0)
    target_std = getattr(dataset, "target_std", 1)

    dataset_unlearn = NewDataset(
        dataset.feature[unlearn_index],
        dataset.target[unlearn_index],
        target_mean,
        target_std,
    )
    dataset_remain = NewDataset(
        dataset.feature[remain_index],
        dataset.target[remain_index],
        target_mean,
        target_std,
    )

    dataset_unlearn.is_scale = dataset.is_scale
    dataset_remain.is_scale = dataset.is_scale
    return dataset_unlearn, dataset_remain, unlearn_index, remain_index


def _influence_path(cfg, model_type, criteria):
    root_dir = OmegaConf.select(cfg, "influence_dir", default=None)
    if root_dir is not None:
        candidate = os.path.join(str(root_dir), f"{model_type}_{criteria}.npy")
        if os.path.exists(candidate):
            return candidate

    model_path = OmegaConf.select(cfg, "model.influence_dir", default=None)
    if model_path is not None:
        model_path = str(model_path)
        if os.path.isfile(model_path):
            return model_path
        candidate = os.path.join(model_path, f"{model_type}_{criteria}.npy")
        if os.path.exists(candidate):
            return candidate

    return os.path.join("influence", f"{model_type}_{criteria}.npy")


def load_influence_file(influence_path, num_samples):
    if not os.path.exists(influence_path):
        raise FileNotFoundError(
            f"Influence file not found: {influence_path}\n"
            "请先运行：python gen_index.py model=conv"
        )
    influences = np.asarray(np.load(influence_path)).reshape(-1)
    if len(influences) != num_samples:
        raise ValueError(
            f"Influence length {len(influences)} does not match num_samples {num_samples}."
        )
    return influences


def select_random_index(num_samples, unlearn_prop, seed):
    rng = np.random.RandomState(seed)
    unlearn_no = max(1, int(num_samples * unlearn_prop))
    return rng.choice(num_samples, size=unlearn_no, replace=False)


def select_helpful_or_harmful_index(influences, unlearn_prop, mode, seed):
    influences = np.asarray(influences).reshape(-1)
    N = len(influences)
    unlearn_no = max(1, int(N * unlearn_prop))
    candidate_no = int(0.31 * N)

    if mode == "helpful":
        candidate_index = np.argsort(influences)[::-1][:candidate_no]
    elif mode == "harmful":
        candidate_index = np.argsort(influences)[:candidate_no]
    else:
        raise ValueError("mode must be helpful or harmful.")

    rng = np.random.RandomState(seed)
    return rng.choice(candidate_index, size=unlearn_no, replace=False).astype(int)


def load_event_time_index(save_dir, num_samples):
    path = os.path.join(save_dir, "unlearn_time_index.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Event time index not found: {path}\n"
            "请先运行：python gen_event_index.py model=conv unlearn_prop=... +event_window=24"
        )
    idx = np.load(path).astype(int)
    idx = idx[(idx >= 0) & (idx < num_samples)]
    return idx


def select_unlearn_index(cfg, index_mode, model_type, num_samples, unlearn_prop, save_dir):
    index_mode = str(index_mode).lower()
    criteria = str(cfg.criteria)
    seed = _safe_int_seed(cfg)
    info = {"index_mode": index_mode, "criteria": criteria}

    if index_mode == "random":
        idx = select_random_index(num_samples, unlearn_prop, seed)
        info["source"] = "random"
        return idx, info

    if index_mode in ["helpful", "harmful"]:
        influence_path = _influence_path(cfg, model_type, criteria)
        influences = load_influence_file(influence_path, num_samples)
        idx = select_helpful_or_harmful_index(influences, unlearn_prop, index_mode, seed)
        info["source"] = f"{index_mode}-{criteria}"
        info["influence_path"] = influence_path
        return idx, info

    if index_mode in ["event", "event_system", "event_mask"]:
        # Repo-style repair is row-level only.
        # event_mask is converted to its selected time rows for alignment.
        idx = load_event_time_index(save_dir, num_samples)
        info["source"] = f"{index_mode}-as-row-level-event-times"
        info["event_time_count"] = int(len(idx))
        return idx, info

    raise ValueError("Supported index_mode: random, helpful, harmful, event, event_system, event_mask")


def solve_reweight_problem(scores_remain, l1_constraint, linf_constraint):
    scores_remain = np.asarray(scores_remain, dtype=float).reshape(-1)
    N = len(scores_remain)

    eps = cp.Variable(N)
    obj = cp.Minimize(cp.scalar_product(eps, scores_remain))
    cons = [
        cp.norm(eps - 1, 1) <= float(l1_constraint) * N,
        cp.norm(eps - 1, np.inf) <= float(linf_constraint),
    ]
    prob = cp.Problem(obj, cons)

    installed = cp.installed_solvers()
    solver_order = [s for s in ["GUROBI", "MOSEK", "CLARABEL", "OSQP", "SCS", "ECOS"] if s in installed]
    last_error = None

    for solver_name in solver_order:
        try:
            prob.solve(verbose=False, solver=solver_name)
            if eps.value is not None:
                return np.asarray(eps.value, dtype=float).reshape(-1), prob.status, solver_name
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"CVXPY failed. Last error: {last_error}")


def make_spo_model(cfg, model, dataset_train):
    operator = Operator(case_config=cfg.case)
    if dataset_train.is_scale:
        mean = dataset_train.target_mean
        std = dataset_train.target_std
    else:
        mean = 0
        std = 1
    spo_model = SPO(trained_model=model, operator=operator, mean=mean, std=std)
    spo_model.eval()
    return spo_model


def evaluate_all(model, dataset_collection, cfg, with_mispatch=False):
    return {
        "mse": evaluate(model, dataset_collection, loss="mse", case_config=cfg.case),
        "mape": evaluate(model, dataset_collection, loss="mape", case_config=cfg.case),
        "cost": evaluate(model, dataset_collection, loss="cost", case_config=cfg.case, with_mispatch=with_mispatch),
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    model_type = str(cfg.model.type)
    print("========== Repo-style Topology PA-MU / TA-MU Alignment ==========")
    print("Current model_type:", model_type)

    if "nn" not in model_type:
        raise ValueError("eval_topo_unchange_repo_style.py 当前阶段只支持 nn_conv / nn_mixer。")

    index_mode = OmegaConf.select(cfg, "index_mode", default=str(cfg.unlearn_mode))
    rho = float(OmegaConf.select(cfg, "rho", default=0.0))
    damping = float(OmegaConf.select(cfg, "damping", default=1e-8))
    criteria = str(cfg.criteria)
    unlearn_prop = float(cfg.unlearn_prop)
    l1_constraints = _get_l1_constraints(cfg)
    linf_constraint = float(OmegaConf.select(cfg, "linf_constraint", default=1.0))
    batch_size = int(cfg.data.batch_size_eval)
    train_loss = str(OmegaConf.select(cfg, "model.train_loss", default="mse"))

    print("index_mode:", index_mode)
    print("criteria:", criteria)
    print("unlearn_prop:", unlearn_prop)
    print("rho:", rho)
    print("damping:", damping)
    print("l1_constraints:", l1_constraints)
    print("linf_constraint:", linf_constraint)
    print("train_loss:", train_loss)

    dataset_train, dataset_test = return_dataset(cfg)
    dataset_core, dataset_sensitive = return_core_datasets(cfg, dataset_to_be_split=dataset_train)
    dataset_train_affine, dataset_test_affine = return_dataset_for_nn_affine(cfg, dataset_sensitive, dataset_test)

    print("Affine train feature shape:", dataset_train_affine.feature.shape)
    print("Affine train target shape:", dataset_train_affine.target.shape)

    num_samples = len(dataset_train_affine)

    # Topology affine model can be ordinary affine if rho=0.
    operator = Operator(case_config=cfg.case)
    A_grid = build_load_bus_adjacency(operator)
    L_grid = build_laplacian(A_grid)

    model_ori, parameter_ori_np = return_topology_affine_model(
        dataset=dataset_train_affine,
        L_grid=L_grid,
        rho=rho,
        damping=damping,
    )
    model_ori.eval()
    parameter_ori = flatten_model(model_ori)

    if np.linalg.norm(parameter_ori - parameter_ori_np) > 1e-4:
        print("[WARN] flatten_model(model_ori) differs from parameter returned by return_topology_affine_model.")
        print("       norm difference:", np.linalg.norm(parameter_ori - parameter_ori_np))

    save_dir = os.path.join(str(cfg.simulation_dir), model_type, "top_fedtamu")
    unlearn_index, index_info = select_unlearn_index(
        cfg=cfg,
        index_mode=index_mode,
        model_type=model_type,
        num_samples=num_samples,
        unlearn_prop=unlearn_prop,
        save_dir=save_dir,
    )
    print("Index info:", index_info)
    print("unlearn rows:", len(unlearn_index))

    dataset_unlearn, dataset_remain, unlearn_index, remain_index = split_dataset_by_index(dataset_train_affine, unlearn_index)

    loader_train = DataLoader(dataset_train_affine, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test_affine, batch_size=batch_size, shuffle=False)
    loader_remain = DataLoader(dataset_remain, batch_size=batch_size, shuffle=False)

    dataset_collection = {"remain": dataset_remain, "unlearn": dataset_unlearn, "test": dataset_test_affine}

    print("getting original performance")
    metrics_original = evaluate_all(model_ori, dataset_collection, cfg, with_mispatch=True)
    print("Original metrics:", metrics_original)

    # Complete unlearning: same style as original eval_unchange.py.
    print("complete unlearning by repository-style stest on remain set")
    module_unlearn = return_module(
        cfg,
        loss_type_dict={"train": train_loss, "test": train_loss},
        loader_dict={"train": loader_remain, "test": loader_remain},
        model=model_ori,
        method="cg",
        watch_progress=False,
    )
    ihvp = module_unlearn.stest(test_idxs=range(len(dataset_remain))).numpy()
    parameter_complete = parameter_ori - ihvp
    model_complete = reconstruct_model(model_ori, parameter_complete)

    metrics_complete = evaluate_all(model_complete, dataset_collection, cfg, with_mispatch=True)
    print("Complete metrics:", metrics_complete)

    # Influence scores: same style as original eval_unchange.py.
    print("calculating per-train-sample gradients")
    module_train = return_module(
        cfg,
        loss_type_dict={"train": train_loss, "test": train_loss},
        loader_dict={"train": loader_remain, "test": loader_train},
        model=model_ori,
        method="cg",
        watch_progress=False,
    )

    start_train = time.time()
    grad_train_all = []
    for i in range(len(dataset_train_affine)):
        grad_train_all.append(module_train.test_loss_grad(test_idxs=[i]).numpy())
    print("time for calculating train grad:", round(time.time() - start_train, 2))

    if criteria != "cost":
        model_test = model_ori
    else:
        model_test = make_spo_model(cfg, model_ori, dataset_train_affine)

    module_test = return_module(
        cfg,
        loss_type_dict={"train": "mse", "test": criteria},
        loader_dict={"train": loader_train, "test": loader_test},
        model=model_test,
        method="cg",
        watch_progress=False,
    )

    start_time = time.time()
    grad_test_ave = module_test.test_loss_grad(test_idxs=range(len(dataset_test_affine)))
    print("time for calculating test grad:", round(time.time() - start_time, 2))

    start_time = time.time()
    M = -module_train.inverse_hvp(vec=grad_test_ave).numpy()
    print("time for calculating M:", round(time.time() - start_time, 2))

    scores = []
    for grad in grad_train_all:
        scores.append(grad @ M)
    scores = np.asarray(scores) / len(dataset_train_affine)

    scores_remain = scores[remain_index]
    scores_unlearn = scores[unlearn_index]

    print("performance change of unlearning estimate:", float(-np.sum(scores_unlearn)))
    print("scores_remain min/max/mean/std:",
          float(np.min(scores_remain)), float(np.max(scores_remain)),
          float(np.mean(scores_remain)), float(np.std(scores_remain)))

    result_log = {
        "repo_style": True,
        "index_mode": str(index_mode),
        "criteria": criteria,
        "unlearn_prop": unlearn_prop,
        "rho": rho,
        "damping": damping,
        "l1_constraints": l1_constraints,
        "linf_constraint": linf_constraint,
        "train_loss": train_loss,
        "index_info": index_info,
        "original": metrics_original,
        "complete": metrics_complete,
        "scores_remain_min": float(np.min(scores_remain)),
        "scores_remain_max": float(np.max(scores_remain)),
        "scores_remain_mean": float(np.mean(scores_remain)),
        "scores_remain_std": float(np.std(scores_remain)),
        "scores_unlearn_sum": float(np.sum(scores_unlearn)),
        "estimated_performance_change_of_unlearning": float(-np.sum(scores_unlearn)),
        "repair": [],
    }

    print("========== Repo-style PA-MU / TA-MU Repair ==========")
    for constraint in l1_constraints:
        print("----------------------------------------------------")
        print("constraint:", constraint)

        eps_remain, status, solver_name = solve_reweight_problem(
            scores_remain=scores_remain,
            l1_constraint=constraint,
            linf_constraint=linf_constraint,
        )
        print("status:", status, "solver:", solver_name)
        print("||eps-1||_1:", float(np.linalg.norm(eps_remain - 1, 1)))
        print("||eps-1||_inf:", float(np.linalg.norm(eps_remain - 1, np.inf)))
        print("eps min/max:", float(np.min(eps_remain)), float(np.max(eps_remain)))

        dataset_remain_with_weight = DatasetWithWeight(dataset_remain, eps_remain)
        loader_remain_with_weight = DataLoader(dataset_remain_with_weight, batch_size=batch_size, shuffle=False)

        module_unlearn_weighted = return_module(
            cfg,
            loss_type_dict={"train": train_loss, "test": train_loss},
            loader_dict={"train": loader_remain_with_weight, "test": loader_remain_with_weight},
            model=model_ori,
            method="cg",
            watch_progress=False,
            with_weight=True,
        )

        ihvp_weighted = module_unlearn_weighted.stest(test_idxs=range(len(dataset_remain))).numpy()
        parameter_repair = parameter_ori - ihvp_weighted
        model_repair = reconstruct_model(model_ori, parameter_repair)

        metrics_repair = evaluate_all(model_repair, dataset_collection, cfg, with_mispatch=False)
        parameter_diff = float(np.linalg.norm(parameter_complete - parameter_repair, 2))

        print("Repaired metrics:", metrics_repair)
        print("parameter difference:", parameter_diff)

        result_log["repair"].append({
            "l1_constraint": float(constraint),
            "status": str(status),
            "solver": str(solver_name),
            "eps_l1": float(np.linalg.norm(eps_remain - 1, 1)),
            "eps_linf": float(np.linalg.norm(eps_remain - 1, np.inf)),
            "eps_min": float(np.min(eps_remain)),
            "eps_max": float(np.max(eps_remain)),
            "metrics": metrics_repair,
            "parameter_diff_to_complete": parameter_diff,
        })

    prop_tag = _format_float_for_path(unlearn_prop)
    rho_tag = _format_float_for_path(rho)
    linf_tag = _format_float_for_path(linf_constraint)

    result_dir = os.path.join(
        str(cfg.simulation_dir),
        model_type,
        "top_fedtamu",
        (
            f"repo_style_{str(index_mode).lower()}_{criteria}"
            f"_prop_{prop_tag}"
            f"_rho_{rho_tag}"
            f"_linf_{linf_tag}"
        ),
    )
    os.makedirs(result_dir, exist_ok=True)

    np.save(os.path.join(result_dir, "parameter_original.npy"), parameter_ori)
    np.save(os.path.join(result_dir, "parameter_complete.npy"), parameter_complete)
    np.save(os.path.join(result_dir, "unlearn_index.npy"), unlearn_index)
    np.save(os.path.join(result_dir, "remain_index.npy"), remain_index)
    np.save(os.path.join(result_dir, "scores.npy"), scores)
    np.save(os.path.join(result_dir, "scores_remain.npy"), scores_remain)
    np.save(os.path.join(result_dir, "scores_unlearn.npy"), scores_unlearn)
    np.save(os.path.join(result_dir, "repair_log.npy"), result_log, allow_pickle=True)

    with open(os.path.join(result_dir, "repair_log.txt"), "w", encoding="utf-8") as f:
        f.write(str(result_log))

    print("Result saved to:", result_dir)


if __name__ == "__main__":
    main()
