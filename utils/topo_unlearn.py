"""
utils/topo_unlearn.py

TOP-FedTAMU++ 阶段 1：拓扑感知 Newton 遗忘模块。

本文件支持两类遗忘方式：

1. index-level unlearning
   - 输入 unlearn_index: shape = [Nu]
   - 表示某些时间样本被遗忘；
   - 对这些时间 t，所有 bus 都参与遗忘；
   - 适用于 random / helpful / event-system。

2. mask-aware unlearning
   - 输入 unlearn_mask: shape = [N, K]
   - 表示 bus-time 级遗忘；
   - unlearn_mask[t, k] = 1 表示只遗忘时间 t 的 bus k；
   - 适用于 event-mask / 局部 bus 污染遗忘。

核心公式：

给定 affine head:

    y_hat[t, k] = theta_k^T x_aug[t]

其中：

    x_aug[t] = [feature[t], 1]

mask-aware 遗忘梯度为：

    g_k = sum_t M[t, k] * (y_hat[t, k] - y[t, k]) * x_aug[t]

其中 M[t, k] 是 unlearn_mask。

mask-aware Hessian 使用 remain 数据：

    H_k = sum_t (1 - M[t, k]) * x_aug[t] x_aug[t]^T

拓扑 Hessian:

    H_topo = blkdiag(H_1, ..., H_K) + rho * kron(L_grid, P)

其中 P 用于控制拓扑正则是否作用到 bias。
默认 regularize_bias=False，即拓扑正则只作用于 weight，不作用于 bias。

注意：
- 这里的 loss scaling 默认使用 sum-loss，与之前已跑通的原型保持一致；
- 如果后续接 PA-MU / TA-MU，需要统一 loss scaling 时，可设置 loss_reduction="mean"。
"""

import numpy as np
import torch

from utils.funcs import ModelNNAffine


def _to_numpy(x):
    """
    将 torch.Tensor / np.ndarray / list 转成 np.ndarray。
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _get_feature_target(dataset):
    """
    从原仓库 MyDataset / affine dataset 中取 feature 和 target。

    预期：
        dataset.feature: [N, d]
        dataset.target:  [N, K]
    """
    if not hasattr(dataset, "feature"):
        raise AttributeError("dataset has no attribute 'feature'.")
    if not hasattr(dataset, "target"):
        raise AttributeError("dataset has no attribute 'target'.")

    feature = _to_numpy(dataset.feature).astype(float)
    target = _to_numpy(dataset.target).astype(float)

    if feature.ndim != 2:
        raise ValueError(
            f"Expected dataset.feature to be 2-D [N, d], got {feature.shape}."
        )

    if target.ndim != 2:
        raise ValueError(
            f"Expected dataset.target to be 2-D [N, K], got {target.shape}."
        )

    if feature.shape[0] != target.shape[0]:
        raise ValueError(
            f"Feature sample number {feature.shape[0]} does not match "
            f"target sample number {target.shape[0]}."
        )

    return feature, target


def _build_augmented_feature(feature):
    """
    为 affine head 构造带 bias 的特征：

        x_aug = [x, 1]

    输入：
        feature: [N, d]

    输出：
        x_aug: [N, d + 1]
    """
    N = feature.shape[0]
    ones = np.ones((N, 1), dtype=float)
    return np.concatenate([feature, ones], axis=1)


def _reshape_parameter(parameter, num_bus):
    """
    将 parameter 展开为：

        B: [K, d_aug]

    其中 K = num_bus，d_aug = feature_dim + 1。
    """
    parameter_np = _to_numpy(parameter).astype(float).reshape(-1)

    if parameter_np.size % num_bus != 0:
        raise ValueError(
            f"Parameter size {parameter_np.size} cannot be divided by "
            f"num_bus {num_bus}."
        )

    return parameter_np.reshape(num_bus, -1)


def _flatten_parameter(parameter_matrix):
    """
    将 [K, d_aug] 参数矩阵展平为 [K * d_aug]。
    """
    return np.asarray(parameter_matrix, dtype=float).reshape(-1)


def predict_affine(feature, parameter, num_bus):
    """
    用 affine head 参数直接预测。

    参数：
        feature:   [N, d]
        parameter: [K * (d + 1)]
        num_bus:   K

    返回：
        pred: [N, K]
    """
    x_aug = _build_augmented_feature(feature)
    B = _reshape_parameter(parameter, num_bus=num_bus)

    if B.shape[1] != x_aug.shape[1]:
        raise ValueError(
            f"Parameter augmented dim {B.shape[1]} does not match "
            f"feature augmented dim {x_aug.shape[1]}."
        )

    pred = x_aug @ B.T
    return pred


def build_system_mask_from_index(num_samples, num_bus, unlearn_index):
    """
    将时间级 unlearn_index 转成系统级 bus-time mask。

    例如 unlearn_index = [10, 11]，num_bus=14，则：

        M[10, :] = 1
        M[11, :] = 1

    这对应 event-system / random / helpful 这类“整行样本遗忘”。
    """
    mask = np.zeros((num_samples, num_bus), dtype=float)

    if unlearn_index is None:
        return mask

    unlearn_index = np.asarray(unlearn_index).astype(int).reshape(-1)
    unlearn_index = unlearn_index[
        (unlearn_index >= 0) & (unlearn_index < num_samples)
    ]

    mask[unlearn_index, :] = 1.0
    return mask


def _validate_unlearn_mask(unlearn_mask, num_samples, num_bus):
    """
    检查 unlearn_mask 的形状，并裁剪到 [0, 1]。
    """
    mask = _to_numpy(unlearn_mask).astype(float)

    if mask.shape != (num_samples, num_bus):
        raise ValueError(
            f"unlearn_mask shape {mask.shape} does not match "
            f"expected shape {(num_samples, num_bus)}."
        )

    # 允许 soft mask，但限制在 [0, 1]
    mask = np.clip(mask, 0.0, 1.0)
    return mask


def compute_masked_unlearn_gradient(
    feature,
    target,
    parameter,
    unlearn_mask,
    loss_reduction="sum",
):
    """
    计算 mask-aware 遗忘梯度。

    对于每个 bus k：

        g_k = sum_t M[t,k] * (y_hat[t,k] - y[t,k]) * x_aug[t]

    参数：
        feature:      [N, d]
        target:       [N, K]
        parameter:    [K * (d + 1)]
        unlearn_mask: [N, K]
        loss_reduction:
            "sum"  与当前已跑通原型一致；
            "mean" 会除以 N*K，后续接原 PA-MU/TA-MU 时可用。

    返回：
        g_flat: [K * (d + 1)]
    """
    feature = np.asarray(feature, dtype=float)
    target = np.asarray(target, dtype=float)

    N, K = target.shape
    x_aug = _build_augmented_feature(feature)
    B = _reshape_parameter(parameter, num_bus=K)

    pred = x_aug @ B.T
    residual = pred - target

    mask = _validate_unlearn_mask(unlearn_mask, N, K)

    d_aug = x_aug.shape[1]
    grad_matrix = np.zeros((K, d_aug), dtype=float)

    for k in range(K):
        # 只对 M[t,k] > 0 的 bus-time 点贡献遗忘梯度。
        weighted_residual_k = mask[:, k] * residual[:, k]
        grad_matrix[k, :] = x_aug.T @ weighted_residual_k

    if loss_reduction == "mean":
        grad_matrix = grad_matrix / float(N * K)
    elif loss_reduction != "sum":
        raise ValueError("loss_reduction must be 'sum' or 'mean'.")

    return _flatten_parameter(grad_matrix)


def compute_masked_block_hessian(
    feature,
    unlearn_mask,
    rho,
    L_grid,
    regularize_bias=False,
    damping=1e-8,
    loss_reduction="sum",
):
    """
    计算 mask-aware 拓扑 Hessian。

    对于每个 bus k：

        H_k = sum_t (1 - M[t,k]) * x_aug[t] x_aug[t]^T

    也就是说：
        - 如果 M[t,k] = 1，时间 t 的 bus k 被遗忘，
          则该 bus 的 Hessian 中不使用这个点；
        - 如果 M[t,j] = 0，其他 bus j 仍保留这个时间点。

    拓扑项：

        H_topo = blkdiag(H_1,...,H_K) + rho * kron(L_grid, P)

    P:
        - regularize_bias=False 时，P 最后一维 bias 为 0；
        - regularize_bias=True 时，P 为单位阵，bias 也被拓扑正则约束。
    """
    feature = np.asarray(feature, dtype=float)
    mask = np.asarray(unlearn_mask, dtype=float)

    N = feature.shape[0]
    K = mask.shape[1]

    x_aug = _build_augmented_feature(feature)
    d_aug = x_aug.shape[1]

    if mask.shape != (N, K):
        raise ValueError(
            f"Mask shape {mask.shape} does not match "
            f"feature sample number {N}."
        )

    H = np.zeros((K * d_aug, K * d_aug), dtype=float)

    for k in range(K):
        # remain_weight[t] = 1 表示该 bus 的该时间样本保留；
        # remain_weight[t] = 0 表示该 bus 的该时间样本被遗忘。
        remain_weight = 1.0 - mask[:, k]
        remain_weight = np.clip(remain_weight, 0.0, 1.0)

        # X^T diag(w) X
        X_weighted = x_aug * remain_weight[:, None]
        H_k = x_aug.T @ X_weighted

        if loss_reduction == "mean":
            H_k = H_k / float(N * K)
        elif loss_reduction != "sum":
            raise ValueError("loss_reduction must be 'sum' or 'mean'.")

        start = k * d_aug
        end = (k + 1) * d_aug
        H[start:end, start:end] = H_k

    # 构造拓扑正则的 P
    P = np.eye(d_aug, dtype=float)

    if not regularize_bias:
        # 最后一维是 bias，不做拓扑耦合
        P[-1, -1] = 0.0

    L_grid_np = _to_numpy(L_grid).astype(float)

    if L_grid_np.shape != (K, K):
        raise ValueError(
            f"L_grid shape {L_grid_np.shape} does not match num_bus {(K, K)}."
        )

    H_topo = H + float(rho) * np.kron(L_grid_np, P)

    # damping 保证可逆和数值稳定
    H_topo = H_topo + float(damping) * np.eye(K * d_aug)

    return H_topo


def make_model_from_parameter(parameter, num_bus):
    """
    根据参数向量构造原仓库 ModelNNAffine。

    为了兼容原 utils.funcs.ModelNNAffine 的构造方式，
    这里优先使用 ModelNNAffine(parameter, no_out=num_bus)。
    """
    parameter_np = np.asarray(parameter, dtype=float).reshape(-1)

    try:
        model = ModelNNAffine(parameter_np, no_out=num_bus)
    except TypeError:
        # 如果原类不接受 keyword no_out，则尝试位置参数
        model = ModelNNAffine(parameter_np, num_bus)

    return model


def topology_sparse_newton_unlearn(
    dataset,
    parameter,
    L_grid,
    rho=1e-3,
    damping=1e-8,
    unlearn_index=None,
    unlearn_mask=None,
    regularize_bias=False,
    loss_reduction="sum",
):
    """
    统一入口：拓扑感知 Newton 遗忘。

    支持两种输入：

    1. unlearn_index:
       shape = [Nu]
       表示整行时间样本遗忘，内部会转成系统级 mask：
           M[t, :] = 1

    2. unlearn_mask:
       shape = [N, K]
       表示 bus-time 级遗忘：
           M[t, k] = 1 只遗忘时间 t 的 bus k

    如果同时传入 unlearn_mask 和 unlearn_index：
       优先使用 unlearn_mask。

    返回：
        model_unlearn:
            ModelNNAffine

        parameter_unlearn:
            np.ndarray, shape = [K * (d + 1)]

        debug:
            dict，包含梯度范数、更新范数、Hessian 条件数等。
    """
    feature, target = _get_feature_target(dataset)
    N, K = target.shape

    parameter_np = _to_numpy(parameter).astype(float).reshape(-1)

    # ------------------------------------------------------------
    # 1. 构造遗忘 mask
    # ------------------------------------------------------------
    if unlearn_mask is not None:
        mask = _validate_unlearn_mask(unlearn_mask, N, K)
        mask_type = "bus_time_mask"
    elif unlearn_index is not None:
        mask = build_system_mask_from_index(
            num_samples=N,
            num_bus=K,
            unlearn_index=unlearn_index,
        )
        mask_type = "system_index"
    else:
        raise ValueError("Either unlearn_mask or unlearn_index must be provided.")

    # ------------------------------------------------------------
    # 2. 计算 mask-aware 遗忘梯度
    # ------------------------------------------------------------
    g_unlearn = compute_masked_unlearn_gradient(
        feature=feature,
        target=target,
        parameter=parameter_np,
        unlearn_mask=mask,
        loss_reduction=loss_reduction,
    )

    # ------------------------------------------------------------
    # 3. 计算 mask-aware 拓扑 Hessian
    # ------------------------------------------------------------
    H_topo = compute_masked_block_hessian(
        feature=feature,
        unlearn_mask=mask,
        rho=rho,
        L_grid=L_grid,
        regularize_bias=regularize_bias,
        damping=damping,
        loss_reduction=loss_reduction,
    )

    # ------------------------------------------------------------
    # 4. 求解 Newton 遗忘步
    #
    # 删除样本的近似更新：
    #     theta_unlearn = theta + H_remain^{-1} * grad_unlearn
    #
    # 符号与之前已跑通的 topology sparse unlearning 保持一致。
    # ------------------------------------------------------------
    delta = np.linalg.solve(H_topo, g_unlearn)

    parameter_unlearn = parameter_np + delta

    model_unlearn = make_model_from_parameter(
        parameter=parameter_unlearn,
        num_bus=K,
    )

    # ------------------------------------------------------------
    # 5. 调试信息
    # ------------------------------------------------------------
    try:
        hessian_condition = float(np.linalg.cond(H_topo))
    except np.linalg.LinAlgError:
        hessian_condition = np.inf

    debug = {
        "mask_type": mask_type,
        "mask_shape": mask.shape,
        "mask_nonzero": int(np.count_nonzero(mask)),
        "gradient_norm": float(np.linalg.norm(g_unlearn)),
        "delta_norm": float(np.linalg.norm(delta)),
        "hessian_condition": hessian_condition,
        "hessian_shape": H_topo.shape,
        "rho": float(rho),
        "damping": float(damping),
        "regularize_bias": bool(regularize_bias),
        "loss_reduction": loss_reduction,
    }

    return model_unlearn, parameter_unlearn, debug


def topology_sparse_newton_unlearn_with_mask(
    dataset,
    parameter,
    L_grid,
    unlearn_mask,
    rho=1e-3,
    damping=1e-8,
    regularize_bias=False,
    loss_reduction="sum",
):
    """
    显式 mask-aware 入口。

    用法：
        model, param, debug = topology_sparse_newton_unlearn_with_mask(
            dataset=dataset_train_affine,
            parameter=parameter_topo,
            L_grid=L_grid,
            unlearn_mask=unlearn_mask,
        )
    """
    return topology_sparse_newton_unlearn(
        dataset=dataset,
        parameter=parameter,
        L_grid=L_grid,
        rho=rho,
        damping=damping,
        unlearn_mask=unlearn_mask,
        regularize_bias=regularize_bias,
        loss_reduction=loss_reduction,
    )


def topology_sparse_newton_unlearn_with_index(
    dataset,
    parameter,
    L_grid,
    unlearn_index,
    rho=1e-3,
    damping=1e-8,
    regularize_bias=False,
    loss_reduction="sum",
):
    """
    显式 index-level 入口。

    用法：
        model, param, debug = topology_sparse_newton_unlearn_with_index(
            dataset=dataset_train_affine,
            parameter=parameter_topo,
            L_grid=L_grid,
            unlearn_index=unlearn_index,
        )
    """
    return topology_sparse_newton_unlearn(
        dataset=dataset,
        parameter=parameter,
        L_grid=L_grid,
        rho=rho,
        damping=damping,
        unlearn_index=unlearn_index,
        regularize_bias=regularize_bias,
        loss_reduction=loss_reduction,
    )


# 为了兼容可能已经在 test_topo_unlearn.py 中使用的名字，
# 提供一个语义更接近原 pipeline 的别名。
def return_topology_unlearn_model(
    dataset,
    parameter,
    L_grid,
    rho=1e-3,
    damping=1e-8,
    unlearn_index=None,
    unlearn_mask=None,
    regularize_bias=False,
    loss_reduction="sum",
):
    """
    兼容式接口。

    优先使用 unlearn_mask；如果没有 mask，则使用 unlearn_index。
    """
    return topology_sparse_newton_unlearn(
        dataset=dataset,
        parameter=parameter,
        L_grid=L_grid,
        rho=rho,
        damping=damping,
        unlearn_index=unlearn_index,
        unlearn_mask=unlearn_mask,
        regularize_bias=regularize_bias,
        loss_reduction=loss_reduction,
    )