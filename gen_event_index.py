# """
# gen_event_index.py
#
# TOP-FedTAMU+ 阶段 1 第四步：事件级遗忘索引生成。
#
# 本脚本不修改原仓库代码，只新增一个事件级 unlearn index 生成流程。
#
# 设计逻辑：
# 1. 读取原仓库 dataset；
# 2. 对 nn_conv / nn_mixer，复用原 func_operation.py 中的
#    return_core_datasets() 和 return_dataset_for_nn_affine()；
# 3. 复用我们已经验证通过的 topology affine head；
# 4. 计算每个 bus、每个时间步的预测残差：
#        score_node[t, k] = |y[t, k] - y_hat[t, k]|
# 5. 聚合为全网时间断面分数：
#        score_time[t] = sum_k score_node[t, k]
# 6. 用滑动窗口形成事件分数：
#        score_event[t] = sum_{tau=t}^{t+W-1} score_time[tau]
# 7. 选择 Top 事件窗口，并合并成 unlearn_time_index；
# 8. 保存结果到：
#        simulation_result/{model_type}/top_fedtamu/
#
# 注意：
# - 第一版使用 residual-only score。
# - 原仓库 evaluate(loss='cost') 当前只返回 cost 数值，不直接返回 dCost/dy_hat。
#   因此 shadow-price score 会在后续增强阶段加入，而不是在这里硬造接口。
# """
#
# import os
# import numpy as np
# import torch
# import hydra
#
# from omegaconf import DictConfig, OmegaConf
#
# from utils import return_dataset
# from utils.optimization import Operator
# from utils.topology import build_load_bus_adjacency, build_laplacian
# from utils.topo_affine import return_topology_affine_model
# from func_operation import (
#     return_core_datasets,
#     return_dataset_for_nn_affine,
# )
#
#
# def compute_node_residual_score(model, dataset):
#     """
#     计算节点-时间预测残差分数。
#
#     参数
#     ----
#     model:
#         ModelNNAffine 模型。
#     dataset:
#         NN-affine 数据集。
#         dataset.feature: [N, d]
#         dataset.target:  [N, K]
#
#     返回
#     ----
#     score_node:
#         shape = [N, K]
#         score_node[t, k] = |y[t, k] - y_hat[t, k]|
#     pred:
#         shape = [N, K]
#     target:
#         shape = [N, K]
#     """
#     model.eval()
#
#     with torch.no_grad():
#         pred = model(dataset.feature)
#
#         # ModelNNAffine 返回的是 tensor；
#         # 这里保留兼容性，如果后续模型返回 tuple，则取最后一个。
#         if isinstance(pred, (tuple, list)):
#             pred = pred[-1]
#
#     pred_np = pred.detach().cpu().numpy()
#     target_np = dataset.target.detach().cpu().numpy()
#
#     score_node = np.abs(target_np - pred_np)
#
#     return score_node, pred_np, target_np
#
#
# def build_event_score(score_time, window):
#     """
#     根据时间断面分数构造滑动窗口事件分数。
#
#     参数
#     ----
#     score_time:
#         shape = [N]
#     window:
#         事件窗口长度。
#
#     返回
#     ----
#     score_event:
#         shape = [N - window + 1]
#     """
#     score_time = np.asarray(score_time).reshape(-1)
#     N = len(score_time)
#
#     if window <= 0:
#         raise ValueError("event window must be positive.")
#
#     if window > N:
#         raise ValueError(
#             f"event window {window} is larger than number of samples {N}."
#         )
#
#     score_event = np.zeros(N - window + 1, dtype=float)
#
#     for start in range(N - window + 1):
#         score_event[start] = np.sum(score_time[start:start + window])
#
#     return score_event
#
#
# def select_top_event_windows(score_event, window, unlearn_prop, total_length):
#     """
#     选择 Top 事件窗口，并合并为时间索引。
#
#     参数
#     ----
#     score_event:
#         shape = [N - window + 1]，每个窗口的事件分数。
#     window:
#         窗口长度。
#     unlearn_prop:
#         希望遗忘的时间样本比例。
#     total_length:
#         原始训练样本数 N。
#
#     返回
#     ----
#     unlearn_time_index:
#         np.ndarray，排序后的时间索引。
#     selected_event_starts:
#         np.ndarray，选择的事件窗口起点。
#     """
#     target_unlearn_no = max(1, int(total_length * unlearn_prop))
#
#     # 事件窗口按分数从高到低排序
#     sorted_starts = np.argsort(score_event)[::-1]
#
#     selected_times = set()
#     selected_event_starts = []
#
#     for start in sorted_starts:
#         selected_event_starts.append(int(start))
#
#         for t in range(start, start + window):
#             selected_times.add(int(t))
#
#         # 合并窗口后达到目标遗忘数量则停止
#         if len(selected_times) >= target_unlearn_no:
#             break
#
#     unlearn_time_index = np.array(sorted(selected_times), dtype=int)
#     selected_event_starts = np.array(selected_event_starts, dtype=int)
#
#     return unlearn_time_index, selected_event_starts
#
#
# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def main(cfg: DictConfig):
#     model_type = cfg.model.type
#
#     print("Current model_type:", model_type)
#
#     if "nn" not in model_type:
#         raise ValueError(
#             "gen_event_index.py 当前阶段只支持 nn_conv / nn_mixer。"
#         )
#
#     # 事件窗口长度。
#     # 如果命令行使用 +event_window=24，则读取该值；
#     # 如果没有提供，则默认 24。
#     event_window = OmegaConf.select(cfg, "event_window", default=24)
#     event_window = int(event_window)
#
#     unlearn_prop = float(cfg.unlearn_prop)
#
#     # 保存路径严格使用原 config.yaml 中已有字段 simulation_dir。
#     save_dir = os.path.join(
#         str(cfg.simulation_dir),
#         str(model_type),
#         "top_fedtamu",
#     )
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 1. 读取原始数据
#     dataset_train, dataset_test = return_dataset(cfg)
#
#     # 2. 复用原仓库 NN-affine 数据转换逻辑
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
#     # 3. 构造 L_grid
#     operator = Operator(case_config=cfg.case)
#     A_grid = build_load_bus_adjacency(operator)
#     L_grid = build_laplacian(A_grid)
#
#     # 4. 训练 topology affine head
#     # 这里使用和前面测试一致的 rho/damping。
#     rho = 1e-3
#     damping = 1e-8
#
#     model_topo, parameter_topo = return_topology_affine_model(
#         dataset=dataset_train_affine,
#         L_grid=L_grid,
#         rho=rho,
#         damping=damping,
#     )
#
#     # 5. 计算 residual-only 节点分数
#     score_node, pred, target = compute_node_residual_score(
#         model=model_topo,
#         dataset=dataset_train_affine,
#     )
#
#     # 6. 聚合为全网时间断面分数
#     # score_time[t] 越大，说明该时间断面整体预测残差越异常。
#     score_time = np.sum(score_node, axis=1)
#
#     # 7. 滑动窗口事件分数
#     score_event = build_event_score(
#         score_time=score_time,
#         window=event_window,
#     )
#
#     # 8. 选择 Top 事件窗口
#     unlearn_time_index, selected_event_starts = select_top_event_windows(
#         score_event=score_event,
#         window=event_window,
#         unlearn_prop=unlearn_prop,
#         total_length=len(dataset_train_affine),
#     )
#
#     # 9. 保存所有中间结果，便于画图和消融
#     np.save(os.path.join(save_dir, "score_node.npy"), score_node)
#     np.save(os.path.join(save_dir, "score_time.npy"), score_time)
#     np.save(os.path.join(save_dir, "score_event.npy"), score_event)
#     np.save(os.path.join(save_dir, "unlearn_time_index.npy"), unlearn_time_index)
#     np.save(os.path.join(save_dir, "selected_event_starts.npy"), selected_event_starts)
#     np.save(os.path.join(save_dir, "pred_train_affine.npy"), pred)
#     np.save(os.path.join(save_dir, "target_train_affine.npy"), target)
#     np.save(os.path.join(save_dir, "parameter_topology_affine.npy"), parameter_topo)
#
#     print("========== Event Index Generation ==========")
#     print("save_dir:", save_dir)
#     print("event_window:", event_window)
#     print("unlearn_prop:", unlearn_prop)
#     print("num_train_samples:", len(dataset_train_affine))
#     print("num_unlearn_time_index:", len(unlearn_time_index))
#     print("num_selected_event_windows:", len(selected_event_starts))
#     print("score_node shape:", score_node.shape)
#     print("score_time shape:", score_time.shape)
#     print("score_event shape:", score_event.shape)
#     print("unlearn_time_index first 20:", unlearn_time_index[:20])
#
#
# if __name__ == "__main__":
#     main()
"""
gen_event_index.py

TOP-FedTAMU++ 阶段 1：生成 residual-event-system 和 residual-event-mask。

本脚本不修改原仓库核心代码，只基于已有 NN-affine 数据和拓扑 affine head
生成两类事件遗忘集合：

1. residual-event-system:
   - 先根据连续时间窗口的全网残差选择异常时间；
   - 对被选中的时间 t，所有 bus 都参与遗忘；
   - 输出 unlearn_mask_system.npy。

2. residual-event-mask:
   - 先根据连续时间窗口的全网残差选择异常时间；
   - 对被选中的时间 t，只选择该时间残差最大的 bus 参与遗忘；
   - 输出 unlearn_mask.npy。

注意：
- 这里不做攻击检测；
- 这里不使用 shadow price；
- residual-event-mask 只是代码内部构造的“局部高残差遗忘请求”；
- 后续 Fed-VJP / 影子价格会用于 TA-MU-cost 的任务感知梯度解耦，而不是用于污染发现。
"""

import os
import numpy as np
import torch
import hydra

from omegaconf import DictConfig, OmegaConf

from utils import return_dataset
from utils.optimization import Operator
from utils.topology import build_load_bus_adjacency, build_laplacian
from utils.topo_affine import return_topology_affine_model

from func_operation import (
    return_core_datasets,
    return_dataset_for_nn_affine,
)


def compute_node_residual_score(model, dataset):
    """
    计算每个时间、每个 bus 的预测残差。

    参数
    ----
    model:
        ModelNNAffine 或兼容模型。
    dataset:
        NN-affine 数据集。
        dataset.feature: [N, d]
        dataset.target:  [N, K]

    返回
    ----
    score_node:
        shape = [N, K]
        score_node[t, k] = |y[t, k] - y_hat[t, k]|

    pred_np:
        shape = [N, K]

    target_np:
        shape = [N, K]
    """
    model.eval()

    with torch.no_grad():
        pred = model(dataset.feature)

        # 保持兼容：如果模型返回 tuple/list，则取最后一个作为预测值。
        if isinstance(pred, (tuple, list)):
            pred = pred[-1]

    pred_np = pred.detach().cpu().numpy()
    target_np = dataset.target.detach().cpu().numpy()

    if pred_np.shape != target_np.shape:
        raise ValueError(
            f"Prediction shape {pred_np.shape} does not match "
            f"target shape {target_np.shape}."
        )

    score_node = np.abs(target_np - pred_np)

    return score_node, pred_np, target_np


def build_event_score(score_time, window):
    """
    根据时间断面分数构造滑动窗口事件分数。

    参数
    ----
    score_time:
        shape = [N]
        每个时间步的全网残差分数。

    window:
        滑动窗口长度。

    返回
    ----
    score_event:
        shape = [N - window + 1]
        score_event[start] = sum(score_time[start:start+window])
    """
    score_time = np.asarray(score_time).reshape(-1)
    N = len(score_time)

    if window <= 0:
        raise ValueError("event_window must be positive.")

    if window > N:
        raise ValueError(
            f"event_window {window} is larger than number of samples {N}."
        )

    score_event = np.zeros(N - window + 1, dtype=float)

    for start in range(N - window + 1):
        score_event[start] = np.sum(score_time[start:start + window])

    return score_event


def select_top_event_windows(score_event, window, unlearn_prop, total_length):
    """
    选择 Top 事件窗口，并合并为时间索引。

    逻辑：
    - 先按窗口分数从高到低排序；
    - 依次加入窗口中的时间点；
    - 合并重叠窗口；
    - 直到达到 unlearn_prop 对应的目标样本数。

    参数
    ----
    score_event:
        shape = [N - window + 1]

    window:
        事件窗口长度。

    unlearn_prop:
        遗忘比例。

    total_length:
        原训练样本数量 N。

    返回
    ----
    unlearn_time_index:
        排序后的事件时间索引。

    selected_event_starts:
        被选择的窗口起点。
    """
    target_unlearn_no = max(1, int(total_length * unlearn_prop))

    sorted_starts = np.argsort(score_event)[::-1]

    selected_times = set()
    selected_event_starts = []

    for start in sorted_starts:
        start = int(start)
        selected_event_starts.append(start)

        for t in range(start, start + window):
            selected_times.add(int(t))

        if len(selected_times) >= target_unlearn_no:
            break

    unlearn_time_index = np.array(sorted(selected_times), dtype=int)
    selected_event_starts = np.array(selected_event_starts, dtype=int)

    return unlearn_time_index, selected_event_starts


def build_event_masks(score_node, unlearn_time_index):
    """
    根据事件时间索引生成两种 bus-time mask。

    1. unlearn_mask_system:
       被选中的时间 t 上，所有 bus 都参与遗忘。
       这是 residual-event-system。

    2. unlearn_mask:
       被选中的时间 t 上，只选择残差最大的 bus 参与遗忘。
       这是 residual-event-mask。

    参数
    ----
    score_node:
        shape = [N, K]
        每个时间、每个 bus 的残差。

    unlearn_time_index:
        shape = [Nu]
        被选中的事件时间索引。

    返回
    ----
    unlearn_mask:
        shape = [N, K]
        局部 bus-time mask。

    unlearn_mask_system:
        shape = [N, K]
        系统级时间事件 mask。

    unlearn_bus_index:
        shape = [N]
        每个事件时间被选中的 bus。非事件时间为 -1。
    """
    N, K = score_node.shape

    unlearn_mask = np.zeros((N, K), dtype=float)
    unlearn_mask_system = np.zeros((N, K), dtype=float)
    unlearn_bus_index = -np.ones(N, dtype=int)

    for t in unlearn_time_index:
        t = int(t)

        if t < 0 or t >= N:
            continue

        # residual-event-system:
        # 事件时间 t 上所有 bus 都参与遗忘。
        unlearn_mask_system[t, :] = 1.0

        # residual-event-mask:
        # 事件时间 t 上，只选择残差最大的 bus。
        suspicious_bus = int(np.argmax(score_node[t]))
        unlearn_mask[t, suspicious_bus] = 1.0
        unlearn_bus_index[t] = suspicious_bus

    return unlearn_mask, unlearn_mask_system, unlearn_bus_index


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    model_type = str(cfg.model.type)

    print("Current model_type:", model_type)

    if "nn" not in model_type:
        raise ValueError(
            "gen_event_index.py 当前阶段只支持 nn_conv / nn_mixer。"
        )

    # event_window 不是原 config.yaml 字段。
    # 命令行可以用 +event_window=24 指定。
    event_window = OmegaConf.select(cfg, "event_window", default=24)
    event_window = int(event_window)

    unlearn_prop = float(cfg.unlearn_prop)

    # 保存路径使用原 config.yaml 中已有的 simulation_dir。
    save_dir = os.path.join(
        str(cfg.simulation_dir),
        model_type,
        "top_fedtamu",
    )
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1. 读取原始数据
    # ------------------------------------------------------------
    dataset_train, dataset_test = return_dataset(cfg)

    # ------------------------------------------------------------
    # 2. 复用原仓库 NN-affine 数据转换逻辑
    # ------------------------------------------------------------
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
    # 3. 构造电网拓扑 L_grid
    # ------------------------------------------------------------
    operator = Operator(case_config=cfg.case)
    A_grid = build_load_bus_adjacency(operator)
    L_grid = build_laplacian(A_grid)

    # ------------------------------------------------------------
    # 4. 训练 topology affine head
    # ------------------------------------------------------------
    rho = 1e-3
    damping = 1e-8

    model_topo, parameter_topo = return_topology_affine_model(
        dataset=dataset_train_affine,
        L_grid=L_grid,
        rho=rho,
        damping=damping,
    )

    # ------------------------------------------------------------
    # 5. 计算节点级 residual score
    # ------------------------------------------------------------
    score_node, pred, target = compute_node_residual_score(
        model=model_topo,
        dataset=dataset_train_affine,
    )

    # ------------------------------------------------------------
    # 6. 聚合为时间级 residual score
    # ------------------------------------------------------------
    score_time = np.sum(score_node, axis=1)

    # ------------------------------------------------------------
    # 7. 构造滑动窗口事件分数
    # ------------------------------------------------------------
    score_event = build_event_score(
        score_time=score_time,
        window=event_window,
    )

    # ------------------------------------------------------------
    # 8. 选择 Top residual event 时间窗口
    # ------------------------------------------------------------
    unlearn_time_index, selected_event_starts = select_top_event_windows(
        score_event=score_event,
        window=event_window,
        unlearn_prop=unlearn_prop,
        total_length=len(dataset_train_affine),
    )

    # ------------------------------------------------------------
    # 9. 生成 bus-time mask
    # ------------------------------------------------------------
    unlearn_mask, unlearn_mask_system, unlearn_bus_index = build_event_masks(
        score_node=score_node,
        unlearn_time_index=unlearn_time_index,
    )

    # ------------------------------------------------------------
    # 10. 保存结果
    # ------------------------------------------------------------
    np.save(os.path.join(save_dir, "score_node.npy"), score_node)
    np.save(os.path.join(save_dir, "score_time.npy"), score_time)
    np.save(os.path.join(save_dir, "score_event.npy"), score_event)

    # 时间级事件 index
    np.save(os.path.join(save_dir, "unlearn_time_index.npy"), unlearn_time_index)
    np.save(os.path.join(save_dir, "selected_event_starts.npy"), selected_event_starts)

    # bus-time mask
    np.save(os.path.join(save_dir, "unlearn_mask.npy"), unlearn_mask)
    np.save(os.path.join(save_dir, "unlearn_mask_system.npy"), unlearn_mask_system)
    np.save(os.path.join(save_dir, "unlearn_bus_index.npy"), unlearn_bus_index)

    # 调试和后续实验中间结果
    np.save(os.path.join(save_dir, "pred_train_affine.npy"), pred)
    np.save(os.path.join(save_dir, "target_train_affine.npy"), target)
    np.save(os.path.join(save_dir, "parameter_topology_affine.npy"), parameter_topo)

    # ------------------------------------------------------------
    # 11. 打印检查信息
    # ------------------------------------------------------------
    print("========== Event Index Generation ==========")
    print("save_dir:", save_dir)
    print("event_window:", event_window)
    print("unlearn_prop:", unlearn_prop)

    print("num_train_samples:", len(dataset_train_affine))
    print("num_unlearn_time_index:", len(unlearn_time_index))
    print("num_selected_event_windows:", len(selected_event_starts))

    print("score_node shape:", score_node.shape)
    print("score_time shape:", score_time.shape)
    print("score_event shape:", score_event.shape)

    print("unlearn_time_index first 20:", unlearn_time_index[:20])

    print("unlearn_mask shape:", unlearn_mask.shape)
    print("unlearn_mask nonzero:", int(np.count_nonzero(unlearn_mask)))

    print("unlearn_mask_system shape:", unlearn_mask_system.shape)
    print(
        "unlearn_mask_system nonzero:",
        int(np.count_nonzero(unlearn_mask_system)),
    )

    if len(unlearn_time_index) > 0:
        print(
            "event-mask selected bus first 20:",
            unlearn_bus_index[unlearn_time_index[:20]],
        )


if __name__ == "__main__":
    main()