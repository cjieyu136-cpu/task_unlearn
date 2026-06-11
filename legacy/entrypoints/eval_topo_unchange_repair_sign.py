# """
# eval_topo_unchange.py
#
# TOP-FedTAMU++ 阶段 1：Topology PA-MU / TA-MU repair prototype。
#
# 本脚本基于已经跑通的 topology affine head 和 mask-aware topology unlearning，
# 实现原论文 PA-MU / TA-MU 风格的重加权修复。
#
# 支持：
#     random
#     helpful
#     harmful
#     event / event_system
#     event_mask
#
# 支持 criteria：
#     mse   -> Topo-PA-MU-mse
#     mape  -> Topo-PA-MU-mape
#     cost  -> Topo-TA-MU-cost
#
# 说明：
# - 本脚本仍然是 centralized NN-affine prototype；
# - 它不是正式联邦版本；
# - Fed-HVP / Fed-VJP 会在阶段 3 实现；
# - 这里的 reweight 先严格遵循原代码 L1 / L∞ 约束；
# - 物理安全时间对齐重加权后续单独做。
# """
#
# import os
# import time
# import numpy as np
# import torch
# import hydra
# import cvxpy as cp
#
# from torch.utils.data import DataLoader
# from omegaconf import DictConfig, OmegaConf
#
# from utils import return_dataset, NewDataset
# from utils.funcs import evaluate
# from utils.optimization import Operator
# from utils.net import SPO
#
# from utils.topology import build_load_bus_adjacency, build_laplacian
# from utils.topo_affine import return_topology_affine_model
# from utils.topo_unlearn import (
#     topology_sparse_newton_unlearn,
#     build_system_mask_from_index,
#     compute_masked_block_hessian,
#     compute_masked_unlearn_gradient,
#     make_model_from_parameter,
# )
#
# from func_operation import (
#     return_core_datasets,
#     return_dataset_for_nn_affine,
#     return_module,
# )
#
#
# # ---------------------------------------------------------------------
# # 基础工具
# # ---------------------------------------------------------------------
# def _to_numpy(x):
#     if isinstance(x, torch.Tensor):
#         return x.detach().cpu().numpy()
#     if isinstance(x, np.ndarray):
#         return x
#     return np.asarray(x)
#
#
# def _safe_int_seed(cfg):
#     return int(cfg.data.random_seed)
#
#
# def _get_l1_constraints(cfg):
#     """
#     原仓库使用 cfg.model.l1_constraints。
#     如果本地配置中没有该字段，则给一个保守默认值。
#     """
#     constraints = OmegaConf.select(cfg, "model.l1_constraints", default=None)
#
#     if constraints is None:
#         return [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
#
#     if isinstance(constraints, (float, int)):
#         return [float(constraints)]
#
#     return [float(x) for x in list(constraints)]
#
#
# def _build_augmented_feature(feature):
#     feature = np.asarray(feature, dtype=float)
#     ones = np.ones((feature.shape[0], 1), dtype=float)
#     return np.concatenate([feature, ones], axis=1)
#
#
# def _reshape_parameter(parameter, num_bus):
#     parameter = _to_numpy(parameter).astype(float).reshape(-1)
#     if parameter.size % num_bus != 0:
#         raise ValueError(
#             f"Parameter size {parameter.size} cannot be divided by num_bus {num_bus}."
#         )
#     return parameter.reshape(num_bus, -1)
#
#
# def split_dataset_by_index(dataset, unlearn_index):
#     """
#     用于 evaluate() 的 row-level remain/unlearn 划分。
#
#     对 event_mask：
#         真正的更新是 bus-time mask；
#         但 evaluate() 的 cost/mse/mape 仍然按完整 y_t 向量评价，
#         所以这里使用被 mask 命中的时间行作为 row-level unlearn set。
#     """
#     N = len(dataset)
#
#     unlearn_index = np.asarray(unlearn_index).astype(int).reshape(-1)
#     unlearn_index = unlearn_index[
#         (unlearn_index >= 0) & (unlearn_index < N)
#     ]
#
#     unlearn_set = set(unlearn_index.tolist())
#     remain_index = np.array(
#         [i for i in range(N) if i not in unlearn_set],
#         dtype=int,
#     )
#
#     target_mean = getattr(dataset, "target_mean", 0)
#     target_std = getattr(dataset, "target_std", 1)
#
#     dataset_unlearn = NewDataset(
#         dataset.feature[unlearn_index],
#         dataset.target[unlearn_index],
#         target_mean,
#         target_std,
#     )
#
#     dataset_remain = NewDataset(
#         dataset.feature[remain_index],
#         dataset.target[remain_index],
#         target_mean,
#         target_std,
#     )
#
#     dataset_unlearn.is_scale = dataset.is_scale
#     dataset_remain.is_scale = dataset.is_scale
#
#     return dataset_unlearn, dataset_remain, unlearn_index, remain_index
#
#
# # ---------------------------------------------------------------------
# # unlearn set / mask 选择逻辑
# # ---------------------------------------------------------------------
# def load_influence_file(influence_path, num_samples):
#     if not os.path.exists(influence_path):
#         raise FileNotFoundError(
#             f"Influence file not found: {influence_path}\n"
#             "请先运行：python gen_index.py model=conv"
#         )
#
#     influences = np.load(influence_path)
#     influences = np.asarray(influences).reshape(-1)
#
#     if len(influences) != num_samples:
#         raise ValueError(
#             f"Influence length {len(influences)} does not match num_samples {num_samples}."
#         )
#
#     return influences
#
#
# def select_random_index(num_samples, unlearn_prop, seed):
#     rng = np.random.RandomState(seed)
#     unlearn_no = max(1, int(num_samples * unlearn_prop))
#     return rng.choice(num_samples, size=unlearn_no, replace=False)
#
#
# def select_helpful_or_harmful_index(influences, unlearn_prop, mode, seed):
#     """
#     对齐原仓库 return_unlearn_datasets() 逻辑：
#     - helpful/harmful 先取 31% candidate；
#     - 再从 candidate 中随机抽取 unlearn_prop 比例。
#     """
#     influences = np.asarray(influences).reshape(-1)
#     N = len(influences)
#
#     unlearn_no = max(1, int(N * unlearn_prop))
#     candidate_no = int(0.31 * N)
#
#     if mode == "helpful":
#         candidate_index = np.argsort(influences)[::-1][:candidate_no]
#     elif mode == "harmful":
#         candidate_index = np.argsort(influences)[:candidate_no]
#     else:
#         raise ValueError("mode must be helpful or harmful.")
#
#     rng = np.random.RandomState(seed)
#     return rng.choice(candidate_index, size=unlearn_no, replace=False).astype(int)
#
#
# def load_event_mask_files(save_dir, num_samples, num_bus):
#     mask_path = os.path.join(save_dir, "unlearn_mask.npy")
#     mask_system_path = os.path.join(save_dir, "unlearn_mask_system.npy")
#     time_index_path = os.path.join(save_dir, "unlearn_time_index.npy")
#
#     if not os.path.exists(mask_path):
#         raise FileNotFoundError(
#             f"Event mask not found: {mask_path}\n"
#             "请先运行：python gen_event_index.py model=conv unlearn_prop=... +event_window=24"
#         )
#
#     if not os.path.exists(mask_system_path):
#         raise FileNotFoundError(
#             f"Event system mask not found: {mask_system_path}\n"
#             "请先运行：python gen_event_index.py model=conv unlearn_prop=... +event_window=24"
#         )
#
#     if not os.path.exists(time_index_path):
#         raise FileNotFoundError(
#             f"Event time index not found: {time_index_path}\n"
#             "请先运行：python gen_event_index.py model=conv unlearn_prop=... +event_window=24"
#         )
#
#     unlearn_mask = np.load(mask_path).astype(float)
#     unlearn_mask_system = np.load(mask_system_path).astype(float)
#     unlearn_time_index = np.load(time_index_path).astype(int)
#
#     if unlearn_mask.shape != (num_samples, num_bus):
#         raise ValueError(
#             f"unlearn_mask shape {unlearn_mask.shape} does not match {(num_samples, num_bus)}."
#         )
#
#     if unlearn_mask_system.shape != (num_samples, num_bus):
#         raise ValueError(
#             f"unlearn_mask_system shape {unlearn_mask_system.shape} does not match "
#             f"{(num_samples, num_bus)}."
#         )
#
#     return unlearn_mask, unlearn_mask_system, unlearn_time_index
#
#
# def select_unlearn_object(
#     cfg,
#     index_mode,
#     model_type,
#     num_samples,
#     num_bus,
#     unlearn_prop,
#     save_dir,
# ):
#     index_mode = str(index_mode).lower()
#     criteria = str(cfg.criteria)
#     seed = _safe_int_seed(cfg)
#
#     info = {
#         "index_mode": index_mode,
#         "criteria": criteria,
#     }
#
#     if index_mode == "random":
#         unlearn_index = select_random_index(
#             num_samples=num_samples,
#             unlearn_prop=unlearn_prop,
#             seed=seed,
#         )
#
#         unlearn_mask = build_system_mask_from_index(
#             num_samples=num_samples,
#             num_bus=num_bus,
#             unlearn_index=unlearn_index,
#         )
#
#         info["source"] = "random"
#         return unlearn_index, unlearn_mask, info
#
#     if index_mode in ["helpful", "harmful"]:
#         influence_path = os.path.join(
#             str(cfg.influence_dir),
#             f"{model_type}_{criteria}.npy",
#         )
#
#         influences = load_influence_file(
#             influence_path=influence_path,
#             num_samples=num_samples,
#         )
#
#         unlearn_index = select_helpful_or_harmful_index(
#             influences=influences,
#             unlearn_prop=unlearn_prop,
#             mode=index_mode,
#             seed=seed,
#         )
#
#         unlearn_mask = build_system_mask_from_index(
#             num_samples=num_samples,
#             num_bus=num_bus,
#             unlearn_index=unlearn_index,
#         )
#
#         info["source"] = f"{index_mode}-{criteria}"
#         info["influence_path"] = influence_path
#         return unlearn_index, unlearn_mask, info
#
#     if index_mode in ["event", "event_system"]:
#         unlearn_mask, unlearn_mask_system, unlearn_time_index = load_event_mask_files(
#             save_dir=save_dir,
#             num_samples=num_samples,
#             num_bus=num_bus,
#         )
#
#         mask_for_update = unlearn_mask_system
#         unlearn_index_for_eval = np.where(mask_for_update.sum(axis=1) > 0)[0].astype(int)
#
#         info["source"] = "residual-event-system"
#         info["event_time_count"] = int(len(unlearn_index_for_eval))
#         info["mask_nonzero"] = int(np.count_nonzero(mask_for_update))
#         return unlearn_index_for_eval, mask_for_update, info
#
#     if index_mode == "event_mask":
#         unlearn_mask, unlearn_mask_system, unlearn_time_index = load_event_mask_files(
#             save_dir=save_dir,
#             num_samples=num_samples,
#             num_bus=num_bus,
#         )
#
#         mask_for_update = unlearn_mask
#         unlearn_index_for_eval = np.where(mask_for_update.sum(axis=1) > 0)[0].astype(int)
#
#         info["source"] = "residual-event-mask"
#         info["event_time_count"] = int(len(unlearn_index_for_eval))
#         info["mask_nonzero"] = int(np.count_nonzero(mask_for_update))
#         return unlearn_index_for_eval, mask_for_update, info
#
#     raise ValueError(
#         f"Unknown index_mode: {index_mode}. "
#         "Supported: random, helpful, harmful, event, event_system, event_mask."
#     )
#
#
# # ---------------------------------------------------------------------
# # gradient / Hessian / score 计算
# # ---------------------------------------------------------------------
# def compute_per_time_remain_gradients(
#     dataset,
#     parameter,
#     remain_mask,
#     loss_reduction="sum",
# ):
#     """
#     计算每个时间步的 remain 梯度 g_t。
#
#     对于每个时间 t：
#
#         g_t = sum_k remain_mask[t,k] * (y_hat[t,k] - y[t,k]) * x_aug[t]
#
#     返回：
#         grad_time:
#             shape = [N, K * d_aug]
#
#     说明：
#     - eps_t 是时间对齐权重；
#     - 所以 PA-MU / TA-MU reweight 是对每个时间 t 的 remain 梯度加权。
#     """
#     feature = _to_numpy(dataset.feature).astype(float)
#     target = _to_numpy(dataset.target).astype(float)
#
#     N, K = target.shape
#     x_aug = _build_augmented_feature(feature)
#     d_aug = x_aug.shape[1]
#
#     B = _reshape_parameter(parameter, num_bus=K)
#     pred = x_aug @ B.T
#     residual = pred - target
#
#     remain_mask = np.asarray(remain_mask, dtype=float)
#
#     if remain_mask.shape != (N, K):
#         raise ValueError(
#             f"remain_mask shape {remain_mask.shape} does not match {(N, K)}."
#         )
#
#     grad_time = np.zeros((N, K * d_aug), dtype=float)
#
#     for t in range(N):
#         grad_matrix_t = np.zeros((K, d_aug), dtype=float)
#
#         for k in range(K):
#             grad_matrix_t[k, :] = (
#                 remain_mask[t, k]
#                 * residual[t, k]
#                 * x_aug[t]
#             )
#
#         if loss_reduction == "mean":
#             grad_matrix_t = grad_matrix_t / float(N * K)
#         elif loss_reduction != "sum":
#             raise ValueError("loss_reduction must be 'sum' or 'mean'.")
#
#         grad_time[t, :] = grad_matrix_t.reshape(-1)
#
#     return grad_time
#
#
# def compute_test_gradient_mse(dataset, parameter, loss_reduction="sum"):
#     """
#     计算 test MSE 对参数的梯度。
#
#     公式：
#         g = sum_{t,k} (y_hat[t,k] - y[t,k]) * x_aug[t]
#     """
#     feature = _to_numpy(dataset.feature).astype(float)
#     target = _to_numpy(dataset.target).astype(float)
#
#     N, K = target.shape
#     x_aug = _build_augmented_feature(feature)
#     d_aug = x_aug.shape[1]
#
#     B = _reshape_parameter(parameter, num_bus=K)
#     pred = x_aug @ B.T
#     residual = pred - target
#
#     grad_matrix = np.zeros((K, d_aug), dtype=float)
#
#     for k in range(K):
#         grad_matrix[k, :] = x_aug.T @ residual[:, k]
#
#     if loss_reduction == "mean":
#         grad_matrix = grad_matrix / float(N * K)
#     elif loss_reduction != "sum":
#         raise ValueError("loss_reduction must be 'sum' or 'mean'.")
#
#     return grad_matrix.reshape(-1)
#
#
# def compute_test_gradient_mape(dataset, parameter, loss_reduction="sum"):
#     """
#     计算 test MAPE 对参数的梯度。
#
#     注意：
#     原 evaluate(loss='mape') 会先 unscale，然后：
#         mean(abs(pred - target) / target) * 100
#
#     这里为 influence / score 计算使用同方向梯度即可。
#     是否乘以 100 不影响 CVXPY 排序和方向太多，但为一致性保留 100。
#     """
#     feature = _to_numpy(dataset.feature).astype(float)
#     target_scaled = _to_numpy(dataset.target).astype(float)
#
#     N, K = target_scaled.shape
#     x_aug = _build_augmented_feature(feature)
#     d_aug = x_aug.shape[1]
#
#     B = _reshape_parameter(parameter, num_bus=K)
#     pred_scaled = x_aug @ B.T
#
#     if dataset.is_scale:
#         mean = _to_numpy(dataset.target_mean).astype(float)
#         std = _to_numpy(dataset.target_std).astype(float)
#         target = target_scaled * std + mean
#         pred = pred_scaled * std + mean
#         chain = std
#     else:
#         target = target_scaled
#         pred = pred_scaled
#         chain = np.ones(K, dtype=float)
#
#     eps = 1e-8
#     sign_term = np.sign(pred - target)
#     denom = np.maximum(target, eps)
#
#     # d(|pred-target| / target) / d(pred_scaled)
#     grad_pred_scaled = sign_term / denom * chain * 100.0
#
#     grad_matrix = np.zeros((K, d_aug), dtype=float)
#
#     for k in range(K):
#         grad_matrix[k, :] = x_aug.T @ grad_pred_scaled[:, k]
#
#     if loss_reduction == "mean":
#         grad_matrix = grad_matrix / float(N * K)
#     elif loss_reduction != "sum":
#         raise ValueError("loss_reduction must be 'sum' or 'mean'.")
#
#     return grad_matrix.reshape(-1)
#
#
# def compute_test_gradient_cost(cfg, model_topo, dataset_train_affine, dataset_test_affine):
#     """
#     使用原仓库 return_module + SPO 计算 criteria=cost 的 test gradient。
#
#     这一步和原 eval_unchange.py 的 cost criterion 路线保持一致：
#         model_test = SPO(model, operator, mean, std)
#         module_test.test_loss_grad(test_idxs=range(len(dataset_test)))
#
#     返回：
#         grad_test_cost: np.ndarray, shape = [num_params]
#     """
#     batch_size = int(cfg.data.batch_size_eval)
#
#     loader_train = DataLoader(
#         dataset_train_affine,
#         batch_size=batch_size,
#         shuffle=False,
#     )
#
#     loader_test = DataLoader(
#         dataset_test_affine,
#         batch_size=batch_size,
#         shuffle=False,
#     )
#
#     operator = Operator(case_config=cfg.case)
#
#     if dataset_train_affine.is_scale:
#         mean = dataset_train_affine.target_mean
#         std = dataset_train_affine.target_std
#     else:
#         mean = 0
#         std = 1
#
#     model_test = SPO(
#         trained_model=model_topo,
#         operator=operator,
#         mean=mean,
#         std=std,
#     )
#
#     model_test.eval()
#
#     module_test = return_module(
#         cfg,
#         loss_type_dict={"train": "mse", "test": "cost"},
#         loader_dict={"train": loader_train, "test": loader_test},
#         model=model_test,
#         method="cg",
#         watch_progress=False,
#     )
#
#     start = time.time()
#     grad_test = module_test.test_loss_grad(
#         test_idxs=range(len(dataset_test_affine))
#     )
#     print("time for calculating topology cost test grad:", round(time.time() - start, 2))
#
#     return _to_numpy(grad_test).astype(float).reshape(-1)
#
#
# def compute_test_gradient(
#     cfg,
#     criteria,
#     model_topo,
#     dataset_train_affine,
#     dataset_test_affine,
#     parameter_topo,
#     loss_reduction="sum",
# ):
#     criteria = str(criteria).lower()
#
#     if criteria == "mse":
#         return compute_test_gradient_mse(
#             dataset=dataset_test_affine,
#             parameter=parameter_topo,
#             loss_reduction=loss_reduction,
#         )
#
#     if criteria == "mape":
#         return compute_test_gradient_mape(
#             dataset=dataset_test_affine,
#             parameter=parameter_topo,
#             loss_reduction=loss_reduction,
#         )
#
#     if criteria == "cost":
#         grad = compute_test_gradient_cost(
#             cfg=cfg,
#             model_topo=model_topo,
#             dataset_train_affine=dataset_train_affine,
#             dataset_test_affine=dataset_test_affine,
#         )
#
#         if grad.shape[0] != len(parameter_topo):
#             raise ValueError(
#                 f"Cost gradient length {grad.shape[0]} does not match "
#                 f"parameter length {len(parameter_topo)}."
#             )
#
#         return grad
#
#     raise ValueError("criteria must be mse, mape, or cost.")
#
#
# def solve_reweight_problem(scores_remain, l1_constraint, linf_constraint):
#     """
#     解原 PA-MU / TA-MU 的 L1 / L∞ reweight 问题：
#
#         min eps^T scores_remain
#
#         s.t.
#             ||eps - 1||_1 <= l1_constraint * N
#             ||eps - 1||_inf <= linf_constraint
#
#     返回：
#         eps_remain
#         status
#     """
#     scores_remain = np.asarray(scores_remain, dtype=float).reshape(-1)
#     N = len(scores_remain)
#
#     eps = cp.Variable(N)
#
#     objective = cp.Minimize(cp.sum(cp.multiply(eps, scores_remain)))
#
#     constraints = [
#         cp.norm(eps - 1, 1) <= float(l1_constraint) * N,
#         cp.norm(eps - 1, "inf") <= float(linf_constraint),
#     ]
#
#     problem = cp.Problem(objective, constraints)
#
#     installed = cp.installed_solvers()
#
#     solver_order = []
#     for solver_name in ["GUROBI", "MOSEK", "CLARABEL", "OSQP", "SCS", "ECOS"]:
#         if solver_name in installed:
#             solver_order.append(solver_name)
#
#     last_error = None
#
#     for solver_name in solver_order:
#         try:
#             problem.solve(solver=solver_name, verbose=False)
#             if eps.value is not None:
#                 return np.asarray(eps.value, dtype=float).reshape(-1), problem.status
#         except Exception as exc:
#             last_error = exc
#
#     raise RuntimeError(
#         f"CVXPY failed to solve reweight problem. Last error: {last_error}"
#     )
#
#
# def apply_topology_repair_update(
#     parameter_topo,
#     H_topo,
#     g_unlearn,
#     grad_time,
#     remain_index,
#     eps_remain,
# ):
#     """
#     根据一阶 influence 近似构造修复后的参数。
#
#     删除 unlearn 对应：
#         + H^{-1} g_unlearn
#
#     remain 样本重加权 eps 对应：
#         - H^{-1} sum_{t in remain} (eps_t - 1) g_t
#
#     因此：
#         theta_repaired
#         =
#         theta
#         + H^{-1} g_unlearn
#         - H^{-1} sum_t (eps_t - 1) g_t
#     """
#     parameter_topo = np.asarray(parameter_topo, dtype=float).reshape(-1)
#
#     weighted_remain_grad = np.zeros_like(parameter_topo)
#
#     for local_pos, t in enumerate(remain_index):
#         weighted_remain_grad += (eps_remain[local_pos] - 1.0) * grad_time[t]
#
#     rhs = g_unlearn - weighted_remain_grad
#
#     delta_repair = np.linalg.solve(H_topo, rhs)
#
#     parameter_repaired = parameter_topo + delta_repair
#
#     return parameter_repaired, delta_repair, weighted_remain_grad
#
#
# def compute_masked_metrics(model, dataset, unlearn_mask):
#     """
#     只评价 M[t,k] = 1 的 bus-time 点。
#     """
#     model.eval()
#
#     with torch.no_grad():
#         pred = model(dataset.feature)
#         if isinstance(pred, (tuple, list)):
#             pred = pred[-1]
#
#     pred_np = _to_numpy(pred).astype(float)
#     target_np = _to_numpy(dataset.target).astype(float)
#     mask = np.asarray(unlearn_mask).astype(bool)
#
#     if dataset.is_scale:
#         mean = _to_numpy(dataset.target_mean).astype(float)
#         std = _to_numpy(dataset.target_std).astype(float)
#         pred_np = pred_np * std + mean
#         target_np = target_np * std + mean
#
#     num_masked = int(np.count_nonzero(mask))
#
#     if num_masked == 0:
#         return {
#             "masked_mse": np.nan,
#             "masked_mape": np.nan,
#             "num_masked_points": 0,
#         }
#
#     diff = pred_np - target_np
#
#     return {
#         "masked_mse": float(np.mean(diff[mask] ** 2)),
#         "masked_mape": float(np.mean(np.abs(diff[mask]) / target_np[mask]) * 100),
#         "num_masked_points": num_masked,
#     }
#
#
# def evaluate_all(model, dataset_collection, cfg):
#     return {
#         "mse": evaluate(model, dataset_collection, loss="mse", case_config=cfg.case),
#         "mape": evaluate(model, dataset_collection, loss="mape", case_config=cfg.case),
#         "cost": evaluate(model, dataset_collection, loss="cost", case_config=cfg.case),
#     }
#
#
# # ---------------------------------------------------------------------
# # 主流程
# # ---------------------------------------------------------------------
# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def main(cfg: DictConfig):
#     model_type = str(cfg.model.type)
#
#     print("========== Topology PA-MU / TA-MU Repair Prototype ==========")
#     print("Current model_type:", model_type)
#
#     if "nn" not in model_type:
#         raise ValueError(
#             "eval_topo_unchange.py 当前阶段只支持 nn_conv / nn_mixer。"
#         )
#
#     index_mode = OmegaConf.select(
#         cfg,
#         "index_mode",
#         default=str(cfg.unlearn_mode),
#     )
#
#     rho = float(OmegaConf.select(cfg, "rho", default=1e-3))
#     damping = float(OmegaConf.select(cfg, "damping", default=1e-8))
#     regularize_bias = bool(OmegaConf.select(cfg, "regularize_bias", default=False))
#     loss_reduction = str(OmegaConf.select(cfg, "loss_reduction", default="sum"))
#
#     criteria = str(cfg.criteria)
#     unlearn_prop = float(cfg.unlearn_prop)
#
#     l1_constraints = _get_l1_constraints(cfg)
#     linf_constraint = float(OmegaConf.select(cfg, "linf_constraint", default=1.0))
#
#     print("rho:", rho)
#     print("damping:", damping)
#     print("regularize_bias:", regularize_bias)
#     print("loss_reduction:", loss_reduction)
#     print("unlearn_prop:", unlearn_prop)
#     print("index_mode:", index_mode)
#     print("criteria:", criteria)
#     print("l1_constraints:", l1_constraints)
#     print("linf_constraint:", linf_constraint)
#
#     # ------------------------------------------------------------
#     # 1. 数据
#     # ------------------------------------------------------------
#     dataset_train, dataset_test = return_dataset(cfg)
#
#     dataset_core, dataset_sensitive = return_core_datasets(
#         cfg,
#         dataset_to_be_split=dataset_train,
#     )
#
#     dataset_train_affine, dataset_test_affine = return_dataset_for_nn_affine(
#         cfg,
#         dataset_sensitive,
#         dataset_test,
#     )
#
#     print("Affine train feature shape:", dataset_train_affine.feature.shape)
#     print("Affine train target shape:", dataset_train_affine.target.shape)
#
#     num_samples = len(dataset_train_affine)
#     num_bus = dataset_train_affine.target.shape[1]
#
#     # ------------------------------------------------------------
#     # 2. 拓扑
#     # ------------------------------------------------------------
#     operator = Operator(case_config=cfg.case)
#     A_grid = build_load_bus_adjacency(operator)
#     L_grid = build_laplacian(A_grid)
#
#     # ------------------------------------------------------------
#     # 3. topology affine head
#     # ------------------------------------------------------------
#     model_topo, parameter_topo = return_topology_affine_model(
#         dataset=dataset_train_affine,
#         L_grid=L_grid,
#         rho=rho,
#         damping=damping,
#     )
#
#     parameter_topo = np.asarray(parameter_topo, dtype=float).reshape(-1)
#
#     # ------------------------------------------------------------
#     # 4. unlearn index / mask
#     # ------------------------------------------------------------
#     save_dir = os.path.join(
#         str(cfg.simulation_dir),
#         model_type,
#         "top_fedtamu",
#     )
#
#     unlearn_index_for_eval, unlearn_mask_for_update, index_info = select_unlearn_object(
#         cfg=cfg,
#         index_mode=index_mode,
#         model_type=model_type,
#         num_samples=num_samples,
#         num_bus=num_bus,
#         unlearn_prop=unlearn_prop,
#         save_dir=save_dir,
#     )
#
#     print("Index info:", index_info)
#     print("row-level unlearn samples:", len(unlearn_index_for_eval))
#     print("mask nonzero:", int(np.count_nonzero(unlearn_mask_for_update)))
#
#     dataset_unlearn, dataset_remain, unlearn_index, remain_index = split_dataset_by_index(
#         dataset_train_affine,
#         unlearn_index_for_eval,
#     )
#
#     dataset_collection = {
#         "remain": dataset_remain,
#         "unlearn": dataset_unlearn,
#         "test": dataset_test_affine,
#     }
#
#     # ------------------------------------------------------------
#     # 5. complete topology unlearning
#     # ------------------------------------------------------------
#     model_complete, parameter_complete, debug_complete = topology_sparse_newton_unlearn(
#         dataset=dataset_train_affine,
#         parameter=parameter_topo,
#         L_grid=L_grid,
#         rho=rho,
#         damping=damping,
#         unlearn_mask=unlearn_mask_for_update,
#         regularize_bias=regularize_bias,
#         loss_reduction=loss_reduction,
#     )
#
#     print("complete unlearning debug:", debug_complete)
#
#     # ------------------------------------------------------------
#     # 6. 构造 H_topo / g_unlearn / grad_time
#     # ------------------------------------------------------------
#     remain_mask = 1.0 - unlearn_mask_for_update
#
#     H_topo = compute_masked_block_hessian(
#         feature=_to_numpy(dataset_train_affine.feature),
#         unlearn_mask=unlearn_mask_for_update,
#         rho=rho,
#         L_grid=L_grid,
#         regularize_bias=regularize_bias,
#         damping=damping,
#         loss_reduction=loss_reduction,
#     )
#
#     g_unlearn = compute_masked_unlearn_gradient(
#         feature=_to_numpy(dataset_train_affine.feature),
#         target=_to_numpy(dataset_train_affine.target),
#         parameter=parameter_topo,
#         unlearn_mask=unlearn_mask_for_update,
#         loss_reduction=loss_reduction,
#     )
#
#     grad_time = compute_per_time_remain_gradients(
#         dataset=dataset_train_affine,
#         parameter=parameter_topo,
#         remain_mask=remain_mask,
#         loss_reduction=loss_reduction,
#     )
#
#     # ------------------------------------------------------------
#     # 7. 计算 criteria 对应 test gradient
#     # ------------------------------------------------------------
#     print("Calculating test gradient for criteria:", criteria)
#
#     grad_test = compute_test_gradient(
#         cfg=cfg,
#         criteria=criteria,
#         model_topo=model_topo,
#         dataset_train_affine=dataset_train_affine,
#         dataset_test_affine=dataset_test_affine,
#         parameter_topo=parameter_topo,
#         loss_reduction=loss_reduction,
#     )
#
#     print("test gradient norm:", float(np.linalg.norm(grad_test)))
#
#     # ------------------------------------------------------------
#     # 8. influence direction M = - H^{-1} grad_test
#     # ------------------------------------------------------------
#     start = time.time()
#     M_vec = -np.linalg.solve(H_topo, grad_test)
#     print("time for calculating topology M:", round(time.time() - start, 2))
#     print("M norm:", float(np.linalg.norm(M_vec)))
#
#     # 每个时间步 remain gradient 的 score
#     scores_all = grad_time @ M_vec
#
#     scores_remain = scores_all[remain_index]
#     scores_unlearn = scores_all[unlearn_index]
#
#     print("score remain mean:", float(np.mean(scores_remain)))
#     print("score unlearn sum:", float(np.sum(scores_unlearn)))
#     print("estimated performance change of unlearning:", float(-np.sum(scores_unlearn)))
#
#     # ------------------------------------------------------------
#     # 9. 原始 / complete unlearn 评价
#     # ------------------------------------------------------------
#     metrics_original = evaluate_all(model_topo, dataset_collection, cfg)
#     metrics_complete = evaluate_all(model_complete, dataset_collection, cfg)
#
#     masked_original = compute_masked_metrics(
#         model=model_topo,
#         dataset=dataset_train_affine,
#         unlearn_mask=unlearn_mask_for_update,
#     )
#
#     masked_complete = compute_masked_metrics(
#         model=model_complete,
#         dataset=dataset_train_affine,
#         unlearn_mask=unlearn_mask_for_update,
#     )
#
#     print("Original metrics:", metrics_original)
#     print("Complete unlearning metrics:", metrics_complete)
#     print("Masked original:", masked_original)
#     print("Masked complete:", masked_complete)
#
#     # ------------------------------------------------------------
#     # 10. PA-MU / TA-MU reweight repair
#     # ------------------------------------------------------------
#     result_log = {
#         "index_mode": str(index_mode),
#         "criteria": criteria,
#         "rho": rho,
#         "damping": damping,
#         "regularize_bias": regularize_bias,
#         "loss_reduction": loss_reduction,
#         "l1_constraints": l1_constraints,
#         "linf_constraint": linf_constraint,
#         "index_info": index_info,
#         "complete_debug": debug_complete,
#         "original": metrics_original,
#         "complete": metrics_complete,
#         "masked_original": masked_original,
#         "masked_complete": masked_complete,
#         "repair": [],
#     }
#
#     print("========== Topology PA-MU / TA-MU Repair ==========")
#
#     for constraint in l1_constraints:
#         print("----------------------------------------------------")
#         print("l1 constraint:", constraint)
#
#         eps_remain, status = solve_reweight_problem(
#             scores_remain=scores_remain,
#             l1_constraint=constraint,
#             linf_constraint=linf_constraint,
#         )
#
#         print("cvx status:", status)
#         print("||eps-1||_1:", float(np.linalg.norm(eps_remain - 1.0, 1)))
#         print("||eps-1||_inf:", float(np.linalg.norm(eps_remain - 1.0, np.inf)))
#         print("eps min/max:", float(np.min(eps_remain)), float(np.max(eps_remain)))
#
#         parameter_repaired, delta_repair, weighted_remain_grad = apply_topology_repair_update(
#             parameter_topo=parameter_topo,
#             H_topo=H_topo,
#             g_unlearn=g_unlearn,
#             grad_time=grad_time,
#             remain_index=remain_index,
#             eps_remain=eps_remain,
#         )
#
#         model_repaired = make_model_from_parameter(
#             parameter=parameter_repaired,
#             num_bus=num_bus,
#         )
#
#         metrics_repaired = evaluate_all(model_repaired, dataset_collection, cfg)
#
#         masked_repaired = compute_masked_metrics(
#             model=model_repaired,
#             dataset=dataset_train_affine,
#             unlearn_mask=unlearn_mask_for_update,
#         )
#
#         parameter_diff_to_complete = float(
#             np.linalg.norm(parameter_repaired - parameter_complete)
#         )
#
#         print("Repaired metrics:", metrics_repaired)
#         print("Masked repaired:", masked_repaired)
#         print("parameter diff to complete:", parameter_diff_to_complete)
#
#         result_log["repair"].append(
#             {
#                 "l1_constraint": float(constraint),
#                 "status": str(status),
#                 "eps_l1": float(np.linalg.norm(eps_remain - 1.0, 1)),
#                 "eps_linf": float(np.linalg.norm(eps_remain - 1.0, np.inf)),
#                 "eps_min": float(np.min(eps_remain)),
#                 "eps_max": float(np.max(eps_remain)),
#                 "metrics": metrics_repaired,
#                 "masked": masked_repaired,
#                 "parameter_diff_to_complete": parameter_diff_to_complete,
#             }
#         )
#
#     # ------------------------------------------------------------
#     # 11. 保存
#     # ------------------------------------------------------------
#     result_dir = os.path.join(
#         str(cfg.simulation_dir),
#         model_type,
#         "top_fedtamu",
#         f"repair_{str(index_mode).lower()}_{criteria}",
#     )
#     os.makedirs(result_dir, exist_ok=True)
#
#     np.save(os.path.join(result_dir, "parameter_topology_original.npy"), parameter_topo)
#     np.save(os.path.join(result_dir, "parameter_topology_complete.npy"), parameter_complete)
#     np.save(os.path.join(result_dir, "unlearn_index_for_eval.npy"), unlearn_index_for_eval)
#     np.save(os.path.join(result_dir, "unlearn_mask_for_update.npy"), unlearn_mask_for_update)
#     np.save(os.path.join(result_dir, "scores_all.npy"), scores_all)
#     np.save(os.path.join(result_dir, "scores_remain.npy"), scores_remain)
#     np.save(os.path.join(result_dir, "scores_unlearn.npy"), scores_unlearn)
#
#     np.save(
#         os.path.join(result_dir, "repair_log.npy"),
#         result_log,
#         allow_pickle=True,
#     )
#
#     with open(os.path.join(result_dir, "repair_log.txt"), "w", encoding="utf-8") as f:
#         f.write(str(result_log))
#
#     print("Result saved to:", result_dir)
#
#
# if __name__ == "__main__":
#     main()

"""
eval_topo_unchange.py

TOP-FedTAMU++ 阶段 1：Topology PA-MU / TA-MU repair prototype。

本脚本基于已经跑通的 topology affine head 和 mask-aware topology unlearning，
实现原论文 PA-MU / TA-MU 风格的重加权修复。

支持：
    random
    helpful
    harmful
    event / event_system
    event_mask

支持 criteria：
    mse   -> Topo-PA-MU-mse
    mape  -> Topo-PA-MU-mape
    cost  -> Topo-TA-MU-cost

新增诊断功能：
    +score_sign=-1 / +score_sign=1

    score_sign=-1:
        M = - H^{-1} grad_test
        与原 eval_unchange.py 的 M 定义保持一致。

    score_sign=1:
        M = + H^{-1} grad_test
        用于检查 cost criterion 下是否存在梯度方向 / 符号不一致问题。

说明：
- 本脚本仍然是 centralized NN-affine prototype；
- 它不是正式联邦版本；
- Fed-HVP / Fed-VJP 会在阶段 3 实现；
- 这里的 reweight 先严格遵循原代码 L1 / L∞ 约束；
- 物理安全时间对齐重加权后续单独做。
"""

import os
import time
import numpy as np
import torch
import hydra
import cvxpy as cp

from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from utils import return_dataset, NewDataset
from utils.funcs import evaluate
from utils.optimization import Operator
from utils.net import SPO

from utils.topology import build_load_bus_adjacency, build_laplacian
from utils.topo_affine import return_topology_affine_model
from utils.topo_unlearn import (
    topology_sparse_newton_unlearn,
    build_system_mask_from_index,
    compute_masked_block_hessian,
    compute_masked_unlearn_gradient,
    make_model_from_parameter,
)

from func_operation import (
    return_core_datasets,
    return_dataset_for_nn_affine,
    return_module,
)


# ---------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _safe_int_seed(cfg):
    """
    原仓库随机种子在 cfg.data.random_seed。
    不使用 cfg.random_seed，避免与原 config.yaml 不一致。
    """
    return int(cfg.data.random_seed)


def _as_float_list(value):
    """
    把 Hydra / OmegaConf 读取出的 list / float 转成 Python float list。
    """
    if value is None:
        return None

    if isinstance(value, (float, int)):
        return [float(value)]

    return [float(x) for x in list(value)]


def _get_l1_constraints(cfg):
    """
    读取 L1 constraint 列表。

    优先级：
    1. 命令行 +l1_constraints=[...]
    2. 原配置 cfg.model.l1_constraints
    3. 默认 [0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.0]
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


def _format_float_for_path(x):
    """
    把浮点数变成路径友好的字符串。
    """
    s = f"{float(x):.6g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def _build_augmented_feature(feature):
    """
    构造带 bias 的 affine feature:

        x_aug = [x, 1]
    """
    feature = np.asarray(feature, dtype=float)
    ones = np.ones((feature.shape[0], 1), dtype=float)
    return np.concatenate([feature, ones], axis=1)


def _reshape_parameter(parameter, num_bus):
    """
    将 flattened affine 参数 reshape 为 [K, d_aug]。

    与原仓库 ModelNNAffine 一致：
        parameter.reshape(no_out, -1)
        weight = parameter[:, :-1]
        bias   = parameter[:, -1]
    """
    parameter = _to_numpy(parameter).astype(float).reshape(-1)

    if parameter.size % num_bus != 0:
        raise ValueError(
            f"Parameter size {parameter.size} cannot be divided by num_bus {num_bus}."
        )

    return parameter.reshape(num_bus, -1)


def split_dataset_by_index(dataset, unlearn_index):
    """
    用于 evaluate() 的 row-level remain/unlearn 划分。

    对 event_mask：
        真正的更新是 bus-time mask；
        但 evaluate() 的 cost/mse/mape 仍然按完整 y_t 向量评价，
        所以这里使用被 mask 命中的时间行作为 row-level unlearn set。
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


# ---------------------------------------------------------------------
# unlearn set / mask 选择逻辑
# ---------------------------------------------------------------------
def load_influence_file(influence_path, num_samples):
    if not os.path.exists(influence_path):
        raise FileNotFoundError(
            f"Influence file not found: {influence_path}\n"
            "请先运行：python gen_index.py model=conv"
        )

    influences = np.load(influence_path)
    influences = np.asarray(influences).reshape(-1)

    if len(influences) != num_samples:
        raise ValueError(
            f"Influence length {len(influences)} does not match num_samples {num_samples}."
        )

    return influences


def select_random_index(num_samples, unlearn_prop, seed):
    """
    随机选择 unlearn_prop 比例的时间样本。
    """
    rng = np.random.RandomState(seed)
    unlearn_no = max(1, int(num_samples * unlearn_prop))
    return rng.choice(num_samples, size=unlearn_no, replace=False)


def select_helpful_or_harmful_index(influences, unlearn_prop, mode, seed):
    """
    对齐原仓库 return_unlearn_datasets() 的 helpful / harmful 逻辑：

    - candidate_no = int(0.31 * N)
    - helpful: 取 influence 最大的 candidate_no 个样本作为候选
    - harmful: 取 influence 最小的 candidate_no 个样本作为候选
    - 再从候选中随机抽取 unlearn_prop * N 个样本
    """
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


def load_event_mask_files(save_dir, num_samples, num_bus):
    """
    读取 gen_event_index.py 生成的 event mask 文件。
    """
    mask_path = os.path.join(save_dir, "unlearn_mask.npy")
    mask_system_path = os.path.join(save_dir, "unlearn_mask_system.npy")
    time_index_path = os.path.join(save_dir, "unlearn_time_index.npy")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Event mask not found: {mask_path}\n"
            "请先运行：python gen_event_index.py model=conv unlearn_prop=... +event_window=24"
        )

    if not os.path.exists(mask_system_path):
        raise FileNotFoundError(
            f"Event system mask not found: {mask_system_path}\n"
            "请先运行：python gen_event_index.py model=conv unlearn_prop=... +event_window=24"
        )

    if not os.path.exists(time_index_path):
        raise FileNotFoundError(
            f"Event time index not found: {time_index_path}\n"
            "请先运行：python gen_event_index.py model=conv unlearn_prop=... +event_window=24"
        )

    unlearn_mask = np.load(mask_path).astype(float)
    unlearn_mask_system = np.load(mask_system_path).astype(float)
    unlearn_time_index = np.load(time_index_path).astype(int)

    if unlearn_mask.shape != (num_samples, num_bus):
        raise ValueError(
            f"unlearn_mask shape {unlearn_mask.shape} does not match {(num_samples, num_bus)}."
        )

    if unlearn_mask_system.shape != (num_samples, num_bus):
        raise ValueError(
            f"unlearn_mask_system shape {unlearn_mask_system.shape} does not match "
            f"{(num_samples, num_bus)}."
        )

    return unlearn_mask, unlearn_mask_system, unlearn_time_index


def select_unlearn_object(
    cfg,
    index_mode,
    model_type,
    num_samples,
    num_bus,
    unlearn_prop,
    save_dir,
):
    """
    返回：
        unlearn_index_for_eval:
            row-level evaluate 使用的时间索引。

        unlearn_mask_for_update:
            真正 topology update 使用的 [N, K] mask。

        info:
            调试信息。
    """
    index_mode = str(index_mode).lower()
    criteria = str(cfg.criteria)
    seed = _safe_int_seed(cfg)

    info = {
        "index_mode": index_mode,
        "criteria": criteria,
    }

    if index_mode == "random":
        unlearn_index = select_random_index(
            num_samples=num_samples,
            unlearn_prop=unlearn_prop,
            seed=seed,
        )

        unlearn_mask = build_system_mask_from_index(
            num_samples=num_samples,
            num_bus=num_bus,
            unlearn_index=unlearn_index,
        )

        info["source"] = "random"
        return unlearn_index, unlearn_mask, info

    if index_mode in ["helpful", "harmful"]:
        influence_path = os.path.join(
            str(cfg.influence_dir),
            f"{model_type}_{criteria}.npy",
        )

        influences = load_influence_file(
            influence_path=influence_path,
            num_samples=num_samples,
        )

        unlearn_index = select_helpful_or_harmful_index(
            influences=influences,
            unlearn_prop=unlearn_prop,
            mode=index_mode,
            seed=seed,
        )

        unlearn_mask = build_system_mask_from_index(
            num_samples=num_samples,
            num_bus=num_bus,
            unlearn_index=unlearn_index,
        )

        info["source"] = f"{index_mode}-{criteria}"
        info["influence_path"] = influence_path
        return unlearn_index, unlearn_mask, info

    if index_mode in ["event", "event_system"]:
        unlearn_mask, unlearn_mask_system, unlearn_time_index = load_event_mask_files(
            save_dir=save_dir,
            num_samples=num_samples,
            num_bus=num_bus,
        )

        mask_for_update = unlearn_mask_system
        unlearn_index_for_eval = np.where(mask_for_update.sum(axis=1) > 0)[0].astype(int)

        info["source"] = "residual-event-system"
        info["event_time_count"] = int(len(unlearn_index_for_eval))
        info["mask_nonzero"] = int(np.count_nonzero(mask_for_update))
        return unlearn_index_for_eval, mask_for_update, info

    if index_mode == "event_mask":
        unlearn_mask, unlearn_mask_system, unlearn_time_index = load_event_mask_files(
            save_dir=save_dir,
            num_samples=num_samples,
            num_bus=num_bus,
        )

        mask_for_update = unlearn_mask
        unlearn_index_for_eval = np.where(mask_for_update.sum(axis=1) > 0)[0].astype(int)

        info["source"] = "residual-event-mask"
        info["event_time_count"] = int(len(unlearn_index_for_eval))
        info["mask_nonzero"] = int(np.count_nonzero(mask_for_update))
        return unlearn_index_for_eval, mask_for_update, info

    raise ValueError(
        f"Unknown index_mode: {index_mode}. "
        "Supported: random, helpful, harmful, event, event_system, event_mask."
    )


# ---------------------------------------------------------------------
# gradient / Hessian / score 计算
# ---------------------------------------------------------------------
def compute_per_time_remain_gradients(
    dataset,
    parameter,
    remain_mask,
    loss_reduction="sum",
):
    """
    计算每个时间步的 remain 梯度 g_t。

    对每个时间 t：

        g_t = sum_k remain_mask[t,k] * (y_hat[t,k] - y[t,k]) * x_aug[t]

    返回：
        grad_time: shape = [N, K * d_aug]

    说明：
    - eps_t 是时间对齐权重；
    - 所以 PA-MU / TA-MU reweight 是对每个时间 t 的 remain 梯度加权。
    """
    feature = _to_numpy(dataset.feature).astype(float)
    target = _to_numpy(dataset.target).astype(float)

    N, K = target.shape
    x_aug = _build_augmented_feature(feature)
    d_aug = x_aug.shape[1]

    B = _reshape_parameter(parameter, num_bus=K)
    pred = x_aug @ B.T
    residual = pred - target

    remain_mask = np.asarray(remain_mask, dtype=float)

    if remain_mask.shape != (N, K):
        raise ValueError(
            f"remain_mask shape {remain_mask.shape} does not match {(N, K)}."
        )

    grad_time = np.zeros((N, K * d_aug), dtype=float)

    for t in range(N):
        grad_matrix_t = np.zeros((K, d_aug), dtype=float)

        for k in range(K):
            grad_matrix_t[k, :] = (
                remain_mask[t, k]
                * residual[t, k]
                * x_aug[t]
            )

        if loss_reduction == "mean":
            grad_matrix_t = grad_matrix_t / float(N * K)
        elif loss_reduction != "sum":
            raise ValueError("loss_reduction must be 'sum' or 'mean'.")

        grad_time[t, :] = grad_matrix_t.reshape(-1)

    return grad_time


def compute_test_gradient_mse(dataset, parameter, loss_reduction="sum"):
    """
    计算 test MSE 对 affine 参数的显式梯度。
    """
    feature = _to_numpy(dataset.feature).astype(float)
    target = _to_numpy(dataset.target).astype(float)

    N, K = target.shape
    x_aug = _build_augmented_feature(feature)
    d_aug = x_aug.shape[1]

    B = _reshape_parameter(parameter, num_bus=K)
    pred = x_aug @ B.T
    residual = pred - target

    grad_matrix = np.zeros((K, d_aug), dtype=float)

    for k in range(K):
        grad_matrix[k, :] = x_aug.T @ residual[:, k]

    if loss_reduction == "mean":
        grad_matrix = grad_matrix / float(N * K)
    elif loss_reduction != "sum":
        raise ValueError("loss_reduction must be 'sum' or 'mean'.")

    return grad_matrix.reshape(-1)


def compute_test_gradient_mape(dataset, parameter, loss_reduction="sum"):
    """
    计算 test MAPE 对 affine 参数的显式梯度。

    原 evaluate(loss='mape') 逻辑：
        mean(abs(pred - target) / target) * 100

    如果数据被 scale，则通过 chain rule 乘以 std。
    """
    feature = _to_numpy(dataset.feature).astype(float)
    target_scaled = _to_numpy(dataset.target).astype(float)

    N, K = target_scaled.shape
    x_aug = _build_augmented_feature(feature)
    d_aug = x_aug.shape[1]

    B = _reshape_parameter(parameter, num_bus=K)
    pred_scaled = x_aug @ B.T

    if dataset.is_scale:
        mean = _to_numpy(dataset.target_mean).astype(float)
        std = _to_numpy(dataset.target_std).astype(float)
        target = target_scaled * std + mean
        pred = pred_scaled * std + mean
        chain = std
    else:
        target = target_scaled
        pred = pred_scaled
        chain = np.ones(K, dtype=float)

    eps = 1e-8
    sign_term = np.sign(pred - target)
    denom = np.maximum(target, eps)

    grad_pred_scaled = sign_term / denom * chain * 100.0

    grad_matrix = np.zeros((K, d_aug), dtype=float)

    for k in range(K):
        grad_matrix[k, :] = x_aug.T @ grad_pred_scaled[:, k]

    if loss_reduction == "mean":
        grad_matrix = grad_matrix / float(N * K)
    elif loss_reduction != "sum":
        raise ValueError("loss_reduction must be 'sum' or 'mean'.")

    return grad_matrix.reshape(-1)


def compute_test_gradient_cost(cfg, model_topo, dataset_train_affine, dataset_test_affine):
    """
    使用原仓库 return_module + SPO 计算 criteria=cost 的 test gradient。

    这一步和原 eval_unchange.py 的 cost criterion 路线保持一致：
        model_test = SPO(model, operator, mean, std)
        module_test.test_loss_grad(test_idxs=range(len(dataset_test)))

    返回：
        grad_test_cost: np.ndarray, shape = [num_params]
    """
    batch_size = int(cfg.data.batch_size_eval)

    loader_train = DataLoader(
        dataset_train_affine,
        batch_size=batch_size,
        shuffle=False,
    )

    loader_test = DataLoader(
        dataset_test_affine,
        batch_size=batch_size,
        shuffle=False,
    )

    operator = Operator(case_config=cfg.case)

    if dataset_train_affine.is_scale:
        mean = dataset_train_affine.target_mean
        std = dataset_train_affine.target_std
    else:
        mean = 0
        std = 1

    model_test = SPO(
        trained_model=model_topo,
        operator=operator,
        mean=mean,
        std=std,
    )

    model_test.eval()

    module_test = return_module(
        cfg,
        loss_type_dict={"train": "mse", "test": "cost"},
        loader_dict={"train": loader_train, "test": loader_test},
        model=model_test,
        method="cg",
        watch_progress=False,
    )

    start = time.time()
    grad_test = module_test.test_loss_grad(
        test_idxs=range(len(dataset_test_affine))
    )
    print("time for calculating topology cost test grad:", round(time.time() - start, 2))

    return _to_numpy(grad_test).astype(float).reshape(-1)


def compute_test_gradient(
    cfg,
    criteria,
    model_topo,
    dataset_train_affine,
    dataset_test_affine,
    parameter_topo,
    loss_reduction="sum",
):
    """
    根据 criteria 计算 test gradient。

    mse/mape:
        使用显式 affine 梯度。

    cost:
        使用原仓库 SPO + return_module 的 cost gradient。
    """
    criteria = str(criteria).lower()

    if criteria == "mse":
        return compute_test_gradient_mse(
            dataset=dataset_test_affine,
            parameter=parameter_topo,
            loss_reduction=loss_reduction,
        )

    if criteria == "mape":
        return compute_test_gradient_mape(
            dataset=dataset_test_affine,
            parameter=parameter_topo,
            loss_reduction=loss_reduction,
        )

    if criteria == "cost":
        grad = compute_test_gradient_cost(
            cfg=cfg,
            model_topo=model_topo,
            dataset_train_affine=dataset_train_affine,
            dataset_test_affine=dataset_test_affine,
        )

        if grad.shape[0] != len(parameter_topo):
            raise ValueError(
                f"Cost gradient length {grad.shape[0]} does not match "
                f"parameter length {len(parameter_topo)}."
            )

        return grad

    raise ValueError("criteria must be mse, mape, or cost.")


def solve_reweight_problem(scores_remain, l1_constraint, linf_constraint):
    """
    解原 PA-MU / TA-MU 的 L1 / L∞ reweight 问题：

        min eps^T scores_remain

        s.t.
            ||eps - 1||_1 <= l1_constraint * N
            ||eps - 1||_inf <= linf_constraint

    返回：
        eps_remain
        status
    """
    scores_remain = np.asarray(scores_remain, dtype=float).reshape(-1)
    N = len(scores_remain)

    eps = cp.Variable(N)

    objective = cp.Minimize(cp.sum(cp.multiply(eps, scores_remain)))

    constraints = [
        cp.norm(eps - 1, 1) <= float(l1_constraint) * N,
        cp.norm(eps - 1, "inf") <= float(linf_constraint),
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
            if eps.value is not None:
                return np.asarray(eps.value, dtype=float).reshape(-1), problem.status
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"CVXPY failed to solve reweight problem. Last error: {last_error}"
    )


def apply_topology_repair_update(
    parameter_topo,
    H_topo,
    g_unlearn,
    grad_time,
    remain_index,
    eps_remain,
    repair_sign=-1.0,
):
    """
    Closed-form 诊断版修复更新。

    删除 unlearn 对应：
        + H^{-1} g_unlearn

    remain 样本重加权项：
        repair_sign * H^{-1} sum_{t in remain} (eps_t - 1) g_t

    repair_sign=-1：
        rhs = g_unlearn - weighted_remain_grad
        这是当前 closed-form 原型之前使用的方向。

    repair_sign=+1：
        rhs = g_unlearn + weighted_remain_grad
        仅用于诊断 remain reweight 项方向是否与原仓库 repo-style stest 流程一致。

    注意：
        这个函数仍是 closed-form 诊断实现，不等价于原仓库
        DatasetWithWeight + return_module(..., with_weight=True) + stest()。
        原仓库对齐请使用 eval_topo_unchange_repo_style.py。
    """
    repair_sign = float(repair_sign)
    if repair_sign not in [-1.0, 1.0]:
        raise ValueError("repair_sign must be -1 or 1.")

    parameter_topo = np.asarray(parameter_topo, dtype=float).reshape(-1)

    weighted_remain_grad = np.zeros_like(parameter_topo)

    for local_pos, t in enumerate(remain_index):
        weighted_remain_grad += (eps_remain[local_pos] - 1.0) * grad_time[t]

    rhs = g_unlearn + repair_sign * weighted_remain_grad

    delta_repair = np.linalg.solve(H_topo, rhs)

    parameter_repaired = parameter_topo + delta_repair

    return parameter_repaired, delta_repair, weighted_remain_grad

def compute_masked_metrics(model, dataset, unlearn_mask):
    """
    只评价 M[t,k] = 1 的 bus-time 点。
    """
    model.eval()

    with torch.no_grad():
        pred = model(dataset.feature)
        if isinstance(pred, (tuple, list)):
            pred = pred[-1]

    pred_np = _to_numpy(pred).astype(float)
    target_np = _to_numpy(dataset.target).astype(float)
    mask = np.asarray(unlearn_mask).astype(bool)

    if dataset.is_scale:
        mean = _to_numpy(dataset.target_mean).astype(float)
        std = _to_numpy(dataset.target_std).astype(float)
        pred_np = pred_np * std + mean
        target_np = target_np * std + mean

    num_masked = int(np.count_nonzero(mask))

    if num_masked == 0:
        return {
            "masked_mse": np.nan,
            "masked_mape": np.nan,
            "num_masked_points": 0,
        }

    diff = pred_np - target_np

    return {
        "masked_mse": float(np.mean(diff[mask] ** 2)),
        "masked_mape": float(np.mean(np.abs(diff[mask]) / target_np[mask]) * 100),
        "num_masked_points": num_masked,
    }


def evaluate_all(model, dataset_collection, cfg):
    """
    同时评价 mse / mape / cost。
    """
    return {
        "mse": evaluate(model, dataset_collection, loss="mse", case_config=cfg.case),
        "mape": evaluate(model, dataset_collection, loss="mape", case_config=cfg.case),
        "cost": evaluate(model, dataset_collection, loss="cost", case_config=cfg.case),
    }


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    model_type = str(cfg.model.type)

    print("========== Topology PA-MU / TA-MU Repair Prototype ==========")
    print("Current model_type:", model_type)

    if "nn" not in model_type:
        raise ValueError(
            "eval_topo_unchange.py 当前阶段只支持 nn_conv / nn_mixer。"
        )

    index_mode = OmegaConf.select(
        cfg,
        "index_mode",
        default=str(cfg.unlearn_mode),
    )

    rho = float(OmegaConf.select(cfg, "rho", default=1e-3))
    damping = float(OmegaConf.select(cfg, "damping", default=1e-8))
    regularize_bias = bool(OmegaConf.select(cfg, "regularize_bias", default=False))
    loss_reduction = str(OmegaConf.select(cfg, "loss_reduction", default="sum"))

    criteria = str(cfg.criteria)
    unlearn_prop = float(cfg.unlearn_prop)

    l1_constraints = _get_l1_constraints(cfg)
    linf_constraint = float(OmegaConf.select(cfg, "linf_constraint", default=1.0))

    # 新增：score_sign
    # -1: 与原 eval_unchange.py 一致
    # +1: 反向诊断
    score_sign = float(OmegaConf.select(cfg, "score_sign", default=-1.0))
    repair_sign = float(OmegaConf.select(cfg, "repair_sign", default=-1.0))

    if score_sign not in [-1.0, 1.0]:
        raise ValueError("score_sign must be -1 or 1.")
    if repair_sign not in [-1.0, 1.0]:
        raise ValueError("repair_sign must be -1 or 1.")

    print("rho:", rho)
    print("damping:", damping)
    print("regularize_bias:", regularize_bias)
    print("loss_reduction:", loss_reduction)
    print("unlearn_prop:", unlearn_prop)
    print("index_mode:", index_mode)
    print("criteria:", criteria)
    print("l1_constraints:", l1_constraints)
    print("linf_constraint:", linf_constraint)
    print("score_sign:", score_sign)
    print("repair_sign:", repair_sign)

    # ------------------------------------------------------------
    # 1. 数据
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

    num_samples = len(dataset_train_affine)
    num_bus = dataset_train_affine.target.shape[1]

    # ------------------------------------------------------------
    # 2. 拓扑
    # ------------------------------------------------------------
    operator = Operator(case_config=cfg.case)
    A_grid = build_load_bus_adjacency(operator)
    L_grid = build_laplacian(A_grid)

    # ------------------------------------------------------------
    # 3. topology affine head
    # ------------------------------------------------------------
    model_topo, parameter_topo = return_topology_affine_model(
        dataset=dataset_train_affine,
        L_grid=L_grid,
        rho=rho,
        damping=damping,
    )

    parameter_topo = np.asarray(parameter_topo, dtype=float).reshape(-1)

    # ------------------------------------------------------------
    # 4. unlearn index / mask
    # ------------------------------------------------------------
    save_dir = os.path.join(
        str(cfg.simulation_dir),
        model_type,
        "top_fedtamu",
    )

    unlearn_index_for_eval, unlearn_mask_for_update, index_info = select_unlearn_object(
        cfg=cfg,
        index_mode=index_mode,
        model_type=model_type,
        num_samples=num_samples,
        num_bus=num_bus,
        unlearn_prop=unlearn_prop,
        save_dir=save_dir,
    )

    print("Index info:", index_info)
    print("row-level unlearn samples:", len(unlearn_index_for_eval))
    print("mask nonzero:", int(np.count_nonzero(unlearn_mask_for_update)))

    dataset_unlearn, dataset_remain, unlearn_index, remain_index = split_dataset_by_index(
        dataset_train_affine,
        unlearn_index_for_eval,
    )

    dataset_collection = {
        "remain": dataset_remain,
        "unlearn": dataset_unlearn,
        "test": dataset_test_affine,
    }

    # ------------------------------------------------------------
    # 5. complete topology unlearning
    # ------------------------------------------------------------
    model_complete, parameter_complete, debug_complete = topology_sparse_newton_unlearn(
        dataset=dataset_train_affine,
        parameter=parameter_topo,
        L_grid=L_grid,
        rho=rho,
        damping=damping,
        unlearn_mask=unlearn_mask_for_update,
        regularize_bias=regularize_bias,
        loss_reduction=loss_reduction,
    )

    print("complete unlearning debug:", debug_complete)

    # ------------------------------------------------------------
    # 6. 构造 H_topo / g_unlearn / grad_time
    # ------------------------------------------------------------
    remain_mask = 1.0 - unlearn_mask_for_update

    H_topo = compute_masked_block_hessian(
        feature=_to_numpy(dataset_train_affine.feature),
        unlearn_mask=unlearn_mask_for_update,
        rho=rho,
        L_grid=L_grid,
        regularize_bias=regularize_bias,
        damping=damping,
        loss_reduction=loss_reduction,
    )

    g_unlearn = compute_masked_unlearn_gradient(
        feature=_to_numpy(dataset_train_affine.feature),
        target=_to_numpy(dataset_train_affine.target),
        parameter=parameter_topo,
        unlearn_mask=unlearn_mask_for_update,
        loss_reduction=loss_reduction,
    )

    grad_time = compute_per_time_remain_gradients(
        dataset=dataset_train_affine,
        parameter=parameter_topo,
        remain_mask=remain_mask,
        loss_reduction=loss_reduction,
    )

    print("H_topo shape:", H_topo.shape)
    print("g_unlearn norm:", float(np.linalg.norm(g_unlearn)))
    print("grad_time shape:", grad_time.shape)

    # ------------------------------------------------------------
    # 7. 计算 criteria 对应 test gradient
    # ------------------------------------------------------------
    print("Calculating test gradient for criteria:", criteria)

    grad_test = compute_test_gradient(
        cfg=cfg,
        criteria=criteria,
        model_topo=model_topo,
        dataset_train_affine=dataset_train_affine,
        dataset_test_affine=dataset_test_affine,
        parameter_topo=parameter_topo,
        loss_reduction=loss_reduction,
    )

    grad_test_norm = float(np.linalg.norm(grad_test))
    print("test gradient norm:", grad_test_norm)

    # ------------------------------------------------------------
    # 8. influence direction M
    #
    # 原论文 / 原 eval_unchange.py:
    #     M = - H^{-1} grad_test
    #
    # 本脚本支持 score_sign:
    #     score_sign=-1: M = -H^{-1} grad_test
    #     score_sign=+1: M = +H^{-1} grad_test
    # ------------------------------------------------------------
    start = time.time()
    inv_hvp_test = np.linalg.solve(H_topo, grad_test)
    M_vec = score_sign * inv_hvp_test
    print("time for calculating topology M:", round(time.time() - start, 2))

    M_norm = float(np.linalg.norm(M_vec))
    print("M norm:", M_norm)

    # 每个时间步 remain gradient 的 score
    scores_all = grad_time @ M_vec

    scores_remain = scores_all[remain_index]
    scores_unlearn = scores_all[unlearn_index]

    print("score statistics:")
    print("  scores_all min/max/mean/std:",
          float(np.min(scores_all)),
          float(np.max(scores_all)),
          float(np.mean(scores_all)),
          float(np.std(scores_all)))
    print("  scores_remain min/max/mean/std:",
          float(np.min(scores_remain)),
          float(np.max(scores_remain)),
          float(np.mean(scores_remain)),
          float(np.std(scores_remain)))
    print("  scores_unlearn min/max/mean/std:",
          float(np.min(scores_unlearn)),
          float(np.max(scores_unlearn)),
          float(np.mean(scores_unlearn)),
          float(np.std(scores_unlearn)))

    print("score unlearn sum:", float(np.sum(scores_unlearn)))
    print("estimated performance change of unlearning:", float(-np.sum(scores_unlearn)))

    # ------------------------------------------------------------
    # 9. 原始 / complete unlearn 评价
    # ------------------------------------------------------------
    metrics_original = evaluate_all(model_topo, dataset_collection, cfg)
    metrics_complete = evaluate_all(model_complete, dataset_collection, cfg)

    masked_original = compute_masked_metrics(
        model=model_topo,
        dataset=dataset_train_affine,
        unlearn_mask=unlearn_mask_for_update,
    )

    masked_complete = compute_masked_metrics(
        model=model_complete,
        dataset=dataset_train_affine,
        unlearn_mask=unlearn_mask_for_update,
    )

    print("Original metrics:", metrics_original)
    print("Complete unlearning metrics:", metrics_complete)
    print("Masked original:", masked_original)
    print("Masked complete:", masked_complete)

    # ------------------------------------------------------------
    # 10. PA-MU / TA-MU reweight repair
    # ------------------------------------------------------------
    result_log = {
        "index_mode": str(index_mode),
        "criteria": criteria,
        "rho": rho,
        "damping": damping,
        "regularize_bias": regularize_bias,
        "loss_reduction": loss_reduction,
        "l1_constraints": l1_constraints,
        "linf_constraint": linf_constraint,
        "score_sign": score_sign,
        "repair_sign": repair_sign,
        "index_info": index_info,
        "complete_debug": debug_complete,
        "grad_test_norm": grad_test_norm,
        "M_norm": M_norm,
        "scores_all_min": float(np.min(scores_all)),
        "scores_all_max": float(np.max(scores_all)),
        "scores_all_mean": float(np.mean(scores_all)),
        "scores_all_std": float(np.std(scores_all)),
        "scores_remain_min": float(np.min(scores_remain)),
        "scores_remain_max": float(np.max(scores_remain)),
        "scores_remain_mean": float(np.mean(scores_remain)),
        "scores_remain_std": float(np.std(scores_remain)),
        "scores_unlearn_sum": float(np.sum(scores_unlearn)),
        "estimated_performance_change_of_unlearning": float(-np.sum(scores_unlearn)),
        "original": metrics_original,
        "complete": metrics_complete,
        "masked_original": masked_original,
        "masked_complete": masked_complete,
        "repair": [],
    }

    print("========== Topology PA-MU / TA-MU Repair ==========")

    for constraint in l1_constraints:
        print("----------------------------------------------------")
        print("l1 constraint:", constraint)

        eps_remain, status = solve_reweight_problem(
            scores_remain=scores_remain,
            l1_constraint=constraint,
            linf_constraint=linf_constraint,
        )

        eps_l1 = float(np.linalg.norm(eps_remain - 1.0, 1))
        eps_linf = float(np.linalg.norm(eps_remain - 1.0, np.inf))
        eps_min = float(np.min(eps_remain))
        eps_max = float(np.max(eps_remain))

        print("cvx status:", status)
        print("||eps-1||_1:", eps_l1)
        print("||eps-1||_inf:", eps_linf)
        print("eps min/max:", eps_min, eps_max)

        parameter_repaired, delta_repair, weighted_remain_grad = apply_topology_repair_update(
            parameter_topo=parameter_topo,
            H_topo=H_topo,
            g_unlearn=g_unlearn,
            grad_time=grad_time,
            remain_index=remain_index,
            eps_remain=eps_remain,
            repair_sign=repair_sign,
        )

        model_repaired = make_model_from_parameter(
            parameter=parameter_repaired,
            num_bus=num_bus,
        )

        metrics_repaired = evaluate_all(model_repaired, dataset_collection, cfg)

        masked_repaired = compute_masked_metrics(
            model=model_repaired,
            dataset=dataset_train_affine,
            unlearn_mask=unlearn_mask_for_update,
        )

        parameter_diff_to_complete = float(
            np.linalg.norm(parameter_repaired - parameter_complete)
        )

        delta_repair_norm = float(np.linalg.norm(delta_repair))
        weighted_remain_grad_norm = float(np.linalg.norm(weighted_remain_grad))

        print("Repaired metrics:", metrics_repaired)
        print("Masked repaired:", masked_repaired)
        print("parameter diff to complete:", parameter_diff_to_complete)
        print("delta_repair norm:", delta_repair_norm)
        print("weighted_remain_grad norm:", weighted_remain_grad_norm)

        result_log["repair"].append(
            {
                "l1_constraint": float(constraint),
                "status": str(status),
                "eps_l1": eps_l1,
                "eps_linf": eps_linf,
                "eps_min": eps_min,
                "eps_max": eps_max,
                "metrics": metrics_repaired,
                "masked": masked_repaired,
                "parameter_diff_to_complete": parameter_diff_to_complete,
                "delta_repair_norm": delta_repair_norm,
                "weighted_remain_grad_norm": weighted_remain_grad_norm,
            }
        )

    # ------------------------------------------------------------
    # 11. 保存
    # ------------------------------------------------------------
    sign_tag = "neg" if score_sign < 0 else "pos"
    repair_tag = "neg" if repair_sign < 0 else "pos"
    linf_tag = _format_float_for_path(linf_constraint)

    prop_tag = _format_float_for_path(unlearn_prop)
    rho_tag = _format_float_for_path(rho)

    result_dir = os.path.join(
        str(cfg.simulation_dir),
        model_type,
        "top_fedtamu",
        (
            f"repair_{str(index_mode).lower()}_{criteria}"
            f"_prop_{prop_tag}"
            f"_rho_{rho_tag}"
            f"_score_{sign_tag}"
            f"_repair_{repair_tag}"
            f"_linf_{linf_tag}"
        ),
    )

    # result_dir = os.path.join(
    #     str(cfg.simulation_dir),
    #     model_type,
    #     "top_fedtamu",
    #     f"repair_{str(index_mode).lower()}_{criteria}_sign_{sign_tag}_linf_{linf_tag}",
    # )
    os.makedirs(result_dir, exist_ok=True)

    np.save(os.path.join(result_dir, "parameter_topology_original.npy"), parameter_topo)
    np.save(os.path.join(result_dir, "parameter_topology_complete.npy"), parameter_complete)
    np.save(os.path.join(result_dir, "unlearn_index_for_eval.npy"), unlearn_index_for_eval)
    np.save(os.path.join(result_dir, "unlearn_mask_for_update.npy"), unlearn_mask_for_update)
    np.save(os.path.join(result_dir, "scores_all.npy"), scores_all)
    np.save(os.path.join(result_dir, "scores_remain.npy"), scores_remain)
    np.save(os.path.join(result_dir, "scores_unlearn.npy"), scores_unlearn)

    np.save(
        os.path.join(result_dir, "repair_log.npy"),
        result_log,
        allow_pickle=True,
    )

    with open(os.path.join(result_dir, "repair_log.txt"), "w", encoding="utf-8") as f:
        f.write(str(result_log))

    print("Result saved to:", result_dir)


if __name__ == "__main__":
    main()