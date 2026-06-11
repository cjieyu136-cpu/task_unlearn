"""
utils/physics_safe_reweight.py

TOP-FedTAMU+ 的时间对齐 + 物理安全重加权。

第一版实现：
1. 全网同一时间步共享 eps_t；
2. L1 约束；
3. eps_min / eps_max；
4. 可选动态安全边界。

后续可继续加入完整 J_f @ delta_y 线性约束。
"""

import numpy as np
import cvxpy as cp


def solve_temporal_reweight(
    q,
    l1_constraint=0.05,
    eps_min=0.95,
    eps_max=1.05,
    solver=None,
):
    """
    求解时间对齐重加权问题：

        min sum_t eps_t * q_t

        s.t.
            ||eps - 1||_1 <= l1_constraint * T
            eps_min <= eps_t <= eps_max

    参数：
        q: shape=[T]，每个时间步的补偿分数
        l1_constraint: L1 约束强度
        eps_min, eps_max: eps 上下界
        solver: 可选 CVXPY solver

    返回：
        eps_value: shape=[T]
    """
    q = np.asarray(q).reshape(-1)
    T = len(q)

    eps = cp.Variable(T)

    objective = cp.Minimize(cp.sum(cp.multiply(eps, q)))

    constraints = [
        cp.norm(eps - 1.0, 1) <= l1_constraint * T,
        eps >= eps_min,
        eps <= eps_max,
    ]

    problem = cp.Problem(objective, constraints)

    if solver is None:
        problem.solve(verbose=False)
    else:
        problem.solve(verbose=False, solver=solver)

    if eps.value is None:
        raise RuntimeError(f"CVXPY failed with status: {problem.status}")

    return eps.value


def compute_dynamic_eps_bounds_from_flow_margin(
    y_hat,
    J_f,
    pf_max,
    base_eps_min=0.95,
    base_eps_max=1.05,
    flow_margin_ratio=0.05,
):
    """
    根据当前预测负荷和 DC 潮流灵敏度，构造动态 eps 上下界。

    这个函数是物理安全约束的轻量实现：
    - 如果某个时间步接近线路极限，则 eps 允许变化更小；
    - 如果系统远离边界，则 eps 可按基础区间变化。

    参数：
        y_hat: shape=[T, K]，预测负荷
        J_f: shape=[no_branch, K]，负荷到线路潮流的线性灵敏度
        pf_max: shape=[no_branch]，线路容量
        base_eps_min, base_eps_max: 基础 eps 范围
        flow_margin_ratio: 至少保留多少线路容量作为安全裕度

    返回：
        eps_min_vec, eps_max_vec: shape=[T]
    """
    T = y_hat.shape[0]

    eps_min_vec = np.full(T, base_eps_min)
    eps_max_vec = np.full(T, base_eps_max)

    # 当前线性潮流估计
    flow_est = y_hat @ J_f.T
    loading_ratio = np.max(np.abs(flow_est) / pf_max.reshape(1, -1), axis=1)

    for t in range(T):
        # 越接近线路极限，允许 eps 偏离 1 越小
        available_margin = max(0.0, 1.0 - loading_ratio[t] - flow_margin_ratio)

        # 将安全裕度转成 eps 最大偏离
        # 这里是保守近似，避免引入复杂非线性约束
        allowed_dev = min(base_eps_max - 1.0, available_margin)

        eps_min_vec[t] = max(base_eps_min, 1.0 - allowed_dev)
        eps_max_vec[t] = min(base_eps_max, 1.0 + allowed_dev)

    return eps_min_vec, eps_max_vec


def solve_temporal_reweight_with_bounds(
    q,
    eps_min_vec,
    eps_max_vec,
    l1_constraint=0.05,
    solver=None,
):
    """
    带时间变化上下界的时间对齐 reweight。

    参数：
        q: shape=[T]
        eps_min_vec: shape=[T]
        eps_max_vec: shape=[T]
    """
    q = np.asarray(q).reshape(-1)
    T = len(q)

    eps = cp.Variable(T)

    objective = cp.Minimize(cp.sum(cp.multiply(eps, q)))

    constraints = [
        cp.norm(eps - 1.0, 1) <= l1_constraint * T,
        eps >= eps_min_vec,
        eps <= eps_max_vec,
    ]

    problem = cp.Problem(objective, constraints)

    if solver is None:
        problem.solve(verbose=False)
    else:
        problem.solve(verbose=False, solver=solver)

    if eps.value is None:
        raise RuntimeError(f"CVXPY failed with status: {problem.status}")

    return eps.value