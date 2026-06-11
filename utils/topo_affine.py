"""
utils/topo_affine.py

TOP-FedTAMU+ 阶段 1：拓扑感知 affine head 解析求解器。

本文件不修改原代码，只新增一个带电网拓扑正则的最后线性层解析训练方法。

原论文/原代码中的 NN-affine 思想：
    1. 先用 CNN 或 MLP-Mixer 提取 feature；
    2. 冻结 feature extractor；
    3. 解析训练最后一层 affine head；
    4. 遗忘阶段只处理最后一层，避免全网络 Hessian 不稳定。

本文件在第 3 步中加入电网拓扑正则：
    min_B 0.5 * ||Y - X_aug @ B.T||_F^2
          + 0.5 * rho * Tr(W.T @ L_grid @ W)

其中：
    X_aug: [N, d+1]，最后一列为 1，对应 bias；
    Y:     [N, K]，K 是 bus 数；
    B:     [K, d+1]，每一行是一个 bus 的 affine 参数；
    W:     [K, d]，不包含 bias；
    L_grid:[K, K]，电网拉普拉斯矩阵。

注意：
    bias 不参与拓扑正则，因为拓扑正则主要约束相邻 bus 的 feature-to-load 映射相似性。
"""

import numpy as np
import torch

from utils.funcs import ModelNNAffine


def _to_numpy(array_like):
    """
    将 torch.Tensor 或 numpy.ndarray 统一转换为 numpy.ndarray。

    参数
    ----
    array_like:
        torch.Tensor 或 np.ndarray。

    返回
    ----
    np.ndarray
    """
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def make_augmented_feature(feature):
    """
    给 feature 增加 bias 列。

    参数
    ----
    feature:
        shape = [N, d] 的 feature。

    返回
    ----
    X_aug:
        shape = [N, d+1]，最后一列为 1。
    """
    X = _to_numpy(feature)

    if X.ndim != 2:
        raise ValueError(
            f"Expected feature shape [N, d], got {X.shape}. "
            "请确认你传入的是 NN-affine 数据集中的 feature，而不是原始 [N, K, F] 特征。"
        )

    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    X_aug = np.concatenate([X, ones], axis=1)

    return X_aug


def fit_topology_affine_parameter(dataset, L_grid, rho=1e-3, damping=1e-8):
    """
    解析求解带拓扑正则的 affine head 参数。

    目标函数：
        min_B 0.5 * ||Y - X_aug @ B.T||_F^2
              + 0.5 * rho * Tr(W.T @ L_grid @ W)

    参数
    ----
    dataset:
        原代码 NewDataset 或类似数据集。
        要求：
            dataset.feature: shape = [N, d]
            dataset.target:  shape = [N, K]
    L_grid:
        电网拉普拉斯矩阵，shape = [K, K]。
    rho:
        拓扑正则系数。
    damping:
        数值稳定项，避免 Hessian 奇异。

    返回
    ----
    parameter:
        shape = [K * (d+1)]。
        展平顺序为：
            [bus0_w..., bus0_b, bus1_w..., bus1_b, ...]
        与原 ModelNNAffine(parameter, no_out=K) 兼容。
    """
    X_aug = make_augmented_feature(dataset.feature)
    Y = _to_numpy(dataset.target)

    if Y.ndim != 2:
        raise ValueError(f"Expected target shape [N, K], got {Y.shape}.")

    N, p = X_aug.shape
    K = Y.shape[1]

    L_grid = np.asarray(L_grid, dtype=float)

    if L_grid.shape != (K, K):
        raise ValueError(
            f"L_grid shape {L_grid.shape} does not match target dimension K={K}."
        )

    # S = X_aug^T X_aug, shape = [p, p]
    S = X_aug.T @ X_aug

    # RHS = Y^T X_aug, shape = [K, p]
    RHS = Y.T @ X_aug

    # P 用于控制拓扑正则只作用在 weight，不作用在 bias。
    # p = d + 1，最后一维是 bias。
    P = np.eye(p)
    P[-1, -1] = 0.0

    # 总 Hessian:
    #   I_K ⊗ S + rho * L_grid ⊗ P
    H = np.kron(np.eye(K), S) + rho * np.kron(L_grid, P)

    # 加 damping，提高数值稳定性。
    H = H + damping * np.eye(H.shape[0])

    # RHS 按 bus 行优先展开，和 ModelNNAffine 的 reshape(no_out, -1) 一致。
    rhs = RHS.reshape(-1)

    parameter = np.linalg.solve(H, rhs)

    return parameter


def return_topology_affine_model(dataset, L_grid, rho=1e-3, damping=1e-8):
    """
    返回一个用拓扑正则解析训练得到的 ModelNNAffine 模型。

    参数
    ----
    dataset:
        NN-affine 数据集。
    L_grid:
        电网拉普拉斯矩阵。
    rho:
        拓扑正则系数。
    damping:
        Hessian 稳定项。

    返回
    ----
    model:
        原代码中的 ModelNNAffine，可以直接用于 evaluate()。
    parameter:
        模型参数，方便后续做 topology-sparse unlearning。
    """
    parameter = fit_topology_affine_parameter(
        dataset=dataset,
        L_grid=L_grid,
        rho=rho,
        damping=damping,
    )

    no_out = dataset.target.shape[1]

    model = ModelNNAffine(
        parameter=parameter,
        no_out=no_out,
    )

    return model, parameter


def topology_regularization_value(parameter, L_grid, no_out):
    """
    计算已训练 affine 参数的拓扑正则值，用于调试和日志记录。

    参数
    ----
    parameter:
        展平后的 affine 参数，shape = [K*(d+1)]。
    L_grid:
        电网拉普拉斯矩阵。
    no_out:
        bus 数 K。

    返回
    ----
    reg_value:
        0.5 * Tr(W.T @ L_grid @ W)
    """
    B = parameter.reshape(no_out, -1)

    # 去掉 bias，只保留 weight
    W = B[:, :-1]

    L_grid = np.asarray(L_grid, dtype=float)

    reg_value = 0.5 * np.trace(W.T @ L_grid @ W)

    return float(reg_value)