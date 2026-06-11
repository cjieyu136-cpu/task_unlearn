"""
utils/topology.py

TOP-FedTAMU+ 阶段 1：电网拓扑工具。

本文件不修改原代码，只新增以下功能：
1. 从原 Operator 中读取支路-节点关联关系；
2. 构造负荷节点邻接矩阵 A_grid；
3. 构造电网拉普拉斯矩阵 L_grid；
4. 计算最短拓扑距离；
5. 构造拓扑遗忘传播权重；
6. 构造 DC 潮流下的负荷-线路潮流灵敏度矩阵 J_f。

设计原则：
- 尽量复用原仓库 utils.optimization.Operator；
- 不改变原始 case_config；
- 不影响原 eval_unlearn.py / eval_unchange.py。
"""

import numpy as np
from collections import deque


def build_load_bus_adjacency(operator):
    """
    根据 Operator 中的支路-节点关联矩阵构造负荷节点邻接矩阵。

    参数
    ----
    operator:
        原仓库 utils.optimization.Operator 实例。

    返回
    ----
    A_grid: np.ndarray
        shape = [no_load, no_load] 的邻接矩阵。
        若两个负荷节点之间有线路相连，则 A_grid[i, j] = 1。
    """
    no_load = operator.no_load

    # 原 Operator.A 通常是支路-节点 incidence matrix，
    # shape = [no_branch, no_bus]。
    branch_bus_incidence = np.asarray(operator.A)

    A_grid = np.zeros((no_load, no_load), dtype=float)

    for branch in branch_bus_incidence:
        # 一条支路通常连接两个 bus，对应 incidence 行中两个非零元素。
        buses = np.where(np.abs(branch) > 0)[0]

        if len(buses) != 2:
            continue

        i, j = int(buses[0]), int(buses[1])

        # 阶段 1 先假设 no_load == no_bus。
        # 原论文 case14 中目标负荷维度为 14，与 no_bus 对齐。
        if i < no_load and j < no_load:
            A_grid[i, j] = 1.0
            A_grid[j, i] = 1.0

    return A_grid


def build_laplacian(A_grid):
    """
    根据邻接矩阵构造图拉普拉斯矩阵 L = D - A。

    参数
    ----
    A_grid: np.ndarray
        shape = [K, K] 的邻接矩阵。

    返回
    ----
    L_grid: np.ndarray
        shape = [K, K] 的拉普拉斯矩阵。
    """
    degree = np.sum(A_grid, axis=1)
    D_grid = np.diag(degree)
    L_grid = D_grid - A_grid
    return L_grid


def shortest_path_distance(A_grid):
    """
    计算无权电网拓扑中的最短路径距离。

    参数
    ----
    A_grid: np.ndarray
        shape = [K, K] 的邻接矩阵。

    返回
    ----
    dist: np.ndarray
        shape = [K, K]，dist[i, j] 表示 i 到 j 的最短拓扑距离。
    """
    K = A_grid.shape[0]
    dist = np.full((K, K), np.inf)

    for source in range(K):
        dist[source, source] = 0.0
        queue = deque([source])

        while queue:
            u = queue.popleft()
            neighbors = np.where(A_grid[u] > 0)[0]

            for v in neighbors:
                if np.isinf(dist[source, v]):
                    dist[source, v] = dist[source, u] + 1.0
                    queue.append(v)

    return dist


def topology_forgetting_weight(dist, source_bus, tau=1.0):
    """
    根据拓扑距离构造遗忘传播权重。

    参数
    ----
    dist: np.ndarray
        shortest_path_distance(A_grid) 的输出。
    source_bus: int
        发生攻击或请求遗忘的源节点。
    tau: float
        衰减系数。tau 越大，遗忘影响传播越远。

    返回
    ----
    omega: np.ndarray
        shape = [K]，每个节点的遗忘强度。
    """
    d = dist[source_bus].copy()

    omega = np.exp(-d / tau)
    omega[np.isinf(d)] = 0.0

    return omega


def build_dc_load_to_flow_jacobian(operator):
    """
    构造 DC 潮流近似下，线路潮流对负荷变化的灵敏度矩阵 J_f。

    线性关系：
        Delta f = J_f Delta y

    原理：
        Bbus * theta = injection
        flow = Bf * theta

    对负荷变化 Delta y，有：
        injection = -Cl * Delta y

    去除参考节点后：
        theta_red = inv(Bbus_red) * injection_red

    因此：
        J_f = - Bf[:, non_ref] @ inv(Bbus_red) @ Cl[non_ref, :]

    参数
    ----
    operator:
        原仓库 utils.optimization.Operator 实例。

    返回
    ----
    J_f: np.ndarray
        shape = [no_branch, no_load]。
    """
    Bbus = np.asarray(operator.Bbus, dtype=float)
    Bf = np.asarray(operator.Bf, dtype=float)
    Cl = np.asarray(operator.Cl, dtype=float)

    no_bus = operator.no_bus

    # ref_index 在原代码中可能是 list / np.ndarray / int。
    ref_index = operator.ref_index
    if isinstance(ref_index, (list, tuple, np.ndarray)):
        ref = int(ref_index[0])
    else:
        ref = int(ref_index)

    non_ref = [i for i in range(no_bus) if i != ref]

    Bbus_red = Bbus[np.ix_(non_ref, non_ref)]
    Cl_red = Cl[non_ref, :]

    # 使用 pinv 而不是 inv，增强数值稳定性。
    Bbus_red_inv = np.linalg.pinv(Bbus_red)

    J_f = -Bf[:, non_ref] @ Bbus_red_inv @ Cl_red

    return np.asarray(J_f, dtype=float)