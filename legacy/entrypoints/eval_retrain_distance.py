"""
eval_retrain_distance.py

TOP-FedTAMU+ 阶段 1：遗忘完整性 / retrain distance 诊断脚本。

用途
----
读取 eval_topo_unchange_repo_style_fixedcrit.py 生成的 repo-style 结果目录，
重新构造：

    1. Original model:
       用完整 affine train set 拟合得到的 topology-aware affine head。

    2. Complete model:
       repo-style complete unlearning 得到的参数，即 result_dir/parameter_complete.npy。

    3. Retrain model:
       只用 remain set 重新拟合 topology-aware affine head。
       这是判断遗忘完整性的“黄金参考”。

    4. Repair model(s):
       按 repo-style PA-MU / TA-MU 的 DatasetWithWeight + stest() 重新计算 repair 参数。
       默认对所有 l1_constraints 都计算，也可以只算一个 l1_select。

输出
----
在 result_dir 下保存：

    retrain_distance_summary.csv
    retrain_distance_summary.npy

重点指标
--------
    param_l2_to_retrain:
        ||theta_method - theta_retrain||_2

    param_rel_l2_to_retrain:
        ||theta_method - theta_retrain||_2 / (||theta_retrain||_2 + eps)

    pred_rmse_to_retrain_test / remain / unlearn:
        方法模型预测与 retrain 模型预测之间的 RMSE。
        这比单看 unlearn cost 更能说明模型是否接近“真正删除后重训”的模型。

运行示例
--------
python eval_retrain_distance.py model=conv unlearn_prop=0.2 +index_mode=helpful +index_criteria=cost +repair_criteria=cost +rho=0.001

只计算 l1=0.15：
python eval_retrain_distance.py model=conv unlearn_prop=0.2 +index_mode=helpful +index_criteria=cost +repair_criteria=cost +rho=0.001 +all_l1=false +l1_select=0.15
"""

import os
import time
import numpy as np
import pandas as pd
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
)
from utils.topology import build_load_bus_adjacency, build_laplacian
from utils.optimization import Operator
from utils.topo_affine import return_topology_affine_model

from func_operation import (
    return_core_datasets,
    return_dataset_for_nn_affine,
    return_module,
)


# ---------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------
def _to_numpy(x):
    """安全转 numpy。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _format_float_for_path(x):
    """与 fixedcrit 脚本保持一致，把 0.001 变成 0p001。"""
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def _as_float_list(value):
    """把 Hydra / OmegaConf 读出的列表转成 Python float list。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in list(value)]


def _get_l1_constraints(cfg):
    """
    与 fixedcrit 脚本保持一致：
    1. 优先读取命令行 +l1_constraints=[...]
    2. 其次读取 cfg.model.l1_constraints
    3. 否则使用默认列表
    """
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
    """
    按 row-level index 切分 dataset。

    注意：
    repo-style repair 是 sample-level / row-level 权重；
    因此这里的 unlearn_index/remain_index 都是时间样本行。
    """
    N = len(dataset)

    unlearn_index = np.asarray(unlearn_index).astype(int).reshape(-1)
    unlearn_index = unlearn_index[
        (unlearn_index >= 0) & (unlearn_index < N)
    ]

    unlearn_set = set(unlearn_index.tolist())
    remain_index = np.array(
        [i for i in range(N) if i not in unlearn_set],
        dtype=int,
    )

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


def evaluate_all(model, dataset_collection, cfg):
    """统一评价 MSE / MAPE / Cost。"""
    return {
        "mse": evaluate(model, dataset_collection, loss="mse", case_config=cfg.case),
        "mape": evaluate(model, dataset_collection, loss="mape", case_config=cfg.case),
        "cost": evaluate(model, dataset_collection, loss="cost", case_config=cfg.case),
    }


def flatten_metrics(prefix, metrics):
    """
    把 evaluate_all() 的嵌套 dict 拉平成一行。
    例如 metrics['mse']['test'] -> prefix_mse_test
    """
    row = {}
    for loss_name, value_dict in metrics.items():
        for split_name, value in value_dict.items():
            row[f"{prefix}_{loss_name}_{split_name}"] = float(value)
    return row


# ---------------------------------------------------------------------
# reweight 求解，与 fixedcrit 版本一致：显式 box + 硬检查
# ---------------------------------------------------------------------
def solve_reweight_problem(scores_remain, l1_constraint, linf_constraint, tol=1e-5):
    """
    解 PA-MU / TA-MU 的 remain-sample reweight 问题。

        min eps^T scores_remain

        s.t.
            ||eps - 1||_1 <= l1_constraint * N
            1 - linf_constraint <= eps_i <= 1 + linf_constraint

    显式 box 与 ||eps-1||_inf <= linf_constraint 等价，
    但比 cp.norm(..., "inf") 在某些 solver 下更稳定。
    """
    scores_remain = np.asarray(scores_remain, dtype=float).reshape(-1)
    N = len(scores_remain)

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
    solver_order = []
    for solver_name in ["GUROBI", "MOSEK", "CLARABEL", "OSQP", "SCS", "ECOS"]:
        if solver_name in installed:
            solver_order.append(solver_name)

    last_error = None

    for solver_name in solver_order:
        try:
            problem.solve(solver=solver_name, verbose=False)

            if eps.value is None:
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
                    f"l1={eps_l1}, linf={eps_linf}, min={eps_min}, max={eps_max}"
                )
                continue

            # 只裁剪极小数值误差，不掩盖明显不可行解。
            eps_value = np.clip(eps_value, lower_bound, upper_bound)
            return eps_value, problem.status, solver_name

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"CVXPY failed or returned invalid eps. Last error: {last_error}"
    )


# ---------------------------------------------------------------------
# 模型预测距离
# ---------------------------------------------------------------------
def predict_numpy(model, dataset):
    """
    得到 unscaled prediction。
    原 evaluate() 在 is_scale=True 时会 unscale；
    这里也保持同样逻辑，便于解释预测距离。
    """
    model.eval()
    with torch.no_grad():
        pred = model(dataset.feature)
        if isinstance(pred, (tuple, list)):
            pred = pred[-1]

    pred_np = _to_numpy(pred).astype(float)

    if dataset.is_scale:
        mean = _to_numpy(dataset.target_mean).astype(float)
        std = _to_numpy(dataset.target_std).astype(float)
        pred_np = pred_np * std + mean

    return pred_np


def prediction_distance(model, model_retrain, dataset):
    """
    计算 method 模型与 retrain 模型在某个 dataset 上的预测距离。
    这里不是预测误差，而是两个模型输出之间的距离。
    """
    pred = predict_numpy(model, dataset)
    pred_ref = predict_numpy(model_retrain, dataset)

    diff = pred - pred_ref

    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    denom = np.maximum(np.abs(pred_ref), 1e-8)
    rel = float(np.mean(np.abs(diff) / denom) * 100.0)

    return {
        "pred_mse_to_retrain": mse,
        "pred_rmse_to_retrain": rmse,
        "pred_mae_to_retrain": mae,
        "pred_mape_to_retrain_percent": rel,
    }


def add_prediction_distances(row, model, model_retrain, dataset_collection):
    """给一行结果增加 train split 上的 prediction-to-retrain 距离。"""
    for split_name, dataset in dataset_collection.items():
        dist = prediction_distance(model, model_retrain, dataset)
        for k, v in dist.items():
            row[f"{split_name}_{k}"] = v
    return row


# ---------------------------------------------------------------------
# 结果目录
# ---------------------------------------------------------------------
def build_result_dir(cfg, model_type, index_mode, index_criteria, repair_criteria,
                     unlearn_prop, rho, linf_constraint):
    """
    与 eval_topo_unchange_repo_style_fixedcrit.py 的保存路径保持一致。
    """
    prop_tag = _format_float_for_path(unlearn_prop)
    rho_tag = _format_float_for_path(rho)
    linf_tag = _format_float_for_path(linf_constraint)

    return os.path.join(
        str(cfg.simulation_dir),
        model_type,
        "top_fedtamu",
        (
            f"repo_style_{str(index_mode).lower()}"
            f"_index_{index_criteria}"
            f"_repair_{repair_criteria}"
            f"_prop_{prop_tag}"
            f"_rho_{rho_tag}"
            f"_linf_{linf_tag}"
        ),
    )


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("========== Retrain Distance / Forgetting Completeness ==========")

    model_type = str(cfg.model.type)
    if "nn" not in model_type:
        raise ValueError("当前脚本只支持 nn_conv / nn_mixer 的 affine-head 实验。")

    index_mode = OmegaConf.select(cfg, "index_mode", default=str(cfg.unlearn_mode))
    index_criteria = str(OmegaConf.select(cfg, "index_criteria", default=str(cfg.criteria)))
    repair_criteria = str(OmegaConf.select(cfg, "repair_criteria", default=str(cfg.criteria)))

    unlearn_prop = float(cfg.unlearn_prop)
    rho = float(OmegaConf.select(cfg, "rho", default=1e-3))
    damping = float(OmegaConf.select(cfg, "damping", default=1e-8))
    linf_constraint = float(OmegaConf.select(cfg, "linf_constraint", default=1.0))

    all_l1 = bool(OmegaConf.select(cfg, "all_l1", default=True))
    l1_select = float(OmegaConf.select(cfg, "l1_select", default=0.15))

    l1_constraints = _get_l1_constraints(cfg)
    if not all_l1:
        l1_constraints = [l1_select]

    print("model_type:", model_type)
    print("index_mode:", index_mode)
    print("index_criteria:", index_criteria)
    print("repair_criteria:", repair_criteria)
    print("unlearn_prop:", unlearn_prop)
    print("rho:", rho)
    print("damping:", damping)
    print("linf_constraint:", linf_constraint)
    print("l1_constraints:", l1_constraints)

    result_dir = build_result_dir(
        cfg=cfg,
        model_type=model_type,
        index_mode=index_mode,
        index_criteria=index_criteria,
        repair_criteria=repair_criteria,
        unlearn_prop=unlearn_prop,
        rho=rho,
        linf_constraint=linf_constraint,
    )

    if not os.path.isdir(result_dir):
        raise FileNotFoundError(
            f"Result directory not found:\n{result_dir}\n"
            "请先运行 eval_topo_unchange_repo_style_fixedcrit.py 生成该目录。"
        )

    print("result_dir:", result_dir)

    # ------------------------------------------------------------
    # 1. 读取数据，与 fixedcrit 脚本完全一致
    # ------------------------------------------------------------
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

    print("Affine train feature shape:", dataset_train_affine.feature.shape)
    print("Affine train target shape:", dataset_train_affine.target.shape)

    # ------------------------------------------------------------
    # 2. 拓扑矩阵
    # ------------------------------------------------------------
    operator = Operator(case_config=cfg.case)
    A_grid = build_load_bus_adjacency(operator)
    L_grid = build_laplacian(A_grid)

    # ------------------------------------------------------------
    # 3. 读取 fixedcrit 已保存的 index / parameter / score
    # ------------------------------------------------------------
    parameter_ori = np.load(os.path.join(result_dir, "parameter_original.npy")).reshape(-1)
    parameter_complete = np.load(os.path.join(result_dir, "parameter_complete.npy")).reshape(-1)

    unlearn_index = np.load(os.path.join(result_dir, "unlearn_index.npy")).astype(int)
    remain_index_saved = np.load(os.path.join(result_dir, "remain_index.npy")).astype(int)
    scores_remain = np.load(os.path.join(result_dir, "scores_remain.npy")).reshape(-1)

    dataset_unlearn, dataset_remain, unlearn_index, remain_index = split_dataset_by_index(
        dataset_train_affine,
        unlearn_index,
    )

    # 防止保存的 remain_index 与当前切分不一致。
    if len(remain_index_saved) != len(remain_index) or not np.all(remain_index_saved == remain_index):
        raise ValueError(
            "remain_index in result_dir does not match recomputed remain_index. "
            "请确认配置、index_mode、unlearn_prop 与 fixedcrit 运行时一致。"
        )

    dataset_collection = {
        "remain": dataset_remain,
        "unlearn": dataset_unlearn,
        "test": dataset_test_affine,
    }

    # ------------------------------------------------------------
    # 4. 构造 original / complete / retrain 模型
    # ------------------------------------------------------------
    # 先用 full train 建一个 model skeleton，然后用 saved parameter 还原。
    model_skeleton, _ = return_topology_affine_model(
        dataset=dataset_train_affine,
        L_grid=L_grid,
        rho=rho,
        damping=damping,
    )

    model_original = reconstruct_model(model_skeleton, parameter_ori)
    model_complete = reconstruct_model(model_skeleton, parameter_complete)

    # Golden retrain：只用 remain set 重新拟合 topology-aware affine head。
    # 这是判断 forgetting completeness 的参考模型。
    start = time.time()
    model_retrain, parameter_retrain = return_topology_affine_model(
        dataset=dataset_remain,
        L_grid=L_grid,
        rho=rho,
        damping=damping,
    )
    parameter_retrain = np.asarray(parameter_retrain, dtype=float).reshape(-1)
    print("time for retrain affine head:", round(time.time() - start, 3), "sec")

    # ------------------------------------------------------------
    # 5. 固定部分：original / complete / retrain
    # ------------------------------------------------------------
    rows = []

    def add_model_row(method_name, model, parameter, l1_constraint=np.nan):
        row = {
            "method": method_name,
            "index_mode": str(index_mode),
            "index_criteria": str(index_criteria),
            "repair_criteria": str(repair_criteria),
            "unlearn_prop": float(unlearn_prop),
            "rho": float(rho),
            "linf_constraint": float(linf_constraint),
            "l1_constraint": float(l1_constraint) if not np.isnan(l1_constraint) else np.nan,
            "param_l2_to_retrain": float(np.linalg.norm(parameter - parameter_retrain, 2)),
            "param_rel_l2_to_retrain": float(
                np.linalg.norm(parameter - parameter_retrain, 2)
                / (np.linalg.norm(parameter_retrain, 2) + 1e-12)
            ),
            "param_l2_to_original": float(np.linalg.norm(parameter - parameter_ori, 2)),
            "param_l2_to_complete": float(np.linalg.norm(parameter - parameter_complete, 2)),
        }

        metrics = evaluate_all(model, dataset_collection, cfg)
        row.update(flatten_metrics("metric", metrics))

        row = add_prediction_distances(
            row=row,
            model=model,
            model_retrain=model_retrain,
            dataset_collection=dataset_collection,
        )

        rows.append(row)

    add_model_row("original", model_original, parameter_ori)
    add_model_row("complete", model_complete, parameter_complete)
    add_model_row("retrain", model_retrain, parameter_retrain)

    # ------------------------------------------------------------
    # 6. 重新计算 repair 参数，并比较到 retrain
    # ------------------------------------------------------------
    batch_size = int(cfg.data.batch_size_eval)
    train_loss = "mse"

    for constraint in l1_constraints:
        print("----------------------------------------------------")
        print("Computing repair distance for l1:", constraint)

        eps_remain, status, solver_name = solve_reweight_problem(
            scores_remain=scores_remain,
            l1_constraint=constraint,
            linf_constraint=linf_constraint,
        )

        dataset_remain_with_weight = DatasetWithWeight(dataset_remain, eps_remain)
        loader_remain_with_weight = DataLoader(
            dataset_remain_with_weight,
            batch_size=batch_size,
            shuffle=False,
        )

        module_unlearn_weighted = return_module(
            cfg,
            loss_type_dict={"train": train_loss, "test": train_loss},
            loader_dict={
                "train": loader_remain_with_weight,
                "test": loader_remain_with_weight,
            },
            model=model_original,
            method="cg",
            watch_progress=False,
            with_weight=True,
        )

        start = time.time()
        ihvp_weighted = module_unlearn_weighted.stest(
            test_idxs=range(len(dataset_remain))
        ).numpy()
        print("time for weighted stest:", round(time.time() - start, 3), "sec")

        parameter_repair = parameter_ori - ihvp_weighted.reshape(-1)
        model_repair = reconstruct_model(model_original, parameter_repair)

        row_before = len(rows)
        add_model_row(
            method_name=f"repair_l1_{constraint}",
            model=model_repair,
            parameter=parameter_repair,
            l1_constraint=constraint,
        )

        # 补充 eps 信息。
        rows[row_before]["solver"] = str(solver_name)
        rows[row_before]["eps_l1"] = float(np.linalg.norm(eps_remain - 1.0, 1))
        rows[row_before]["eps_linf"] = float(np.linalg.norm(eps_remain - 1.0, np.inf))
        rows[row_before]["eps_min"] = float(np.min(eps_remain))
        rows[row_before]["eps_max"] = float(np.max(eps_remain))
        rows[row_before]["status"] = str(status)

    # ------------------------------------------------------------
    # 7. 保存
    # ------------------------------------------------------------
    df = pd.DataFrame(rows)

    csv_path = os.path.join(result_dir, "retrain_distance_summary.csv")
    npy_path = os.path.join(result_dir, "retrain_distance_summary.npy")

    df.to_csv(csv_path, index=False)
    np.save(npy_path, rows, allow_pickle=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    show_cols = [
        "method",
        "l1_constraint",
        "param_l2_to_retrain",
        "param_rel_l2_to_retrain",
        "remain_pred_rmse_to_retrain",
        "unlearn_pred_rmse_to_retrain",
        "test_pred_rmse_to_retrain",
        "metric_mse_test",
        "metric_mape_test",
        "metric_cost_test",
        "metric_cost_unlearn",
        "eps_linf",
        "eps_min",
        "eps_max",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    print("\n========== Retrain Distance Summary ==========")
    print(df[show_cols].to_string(index=False))
    print("\nSaved to:", csv_path)


if __name__ == "__main__":
    main()
