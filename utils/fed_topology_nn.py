"""
utils/fed_topology_nn.py

Topology-aware local-input NN federation helpers.

This module keeps the repository's "frozen NN backbone + affine repair" idea,
but replaces the centralized full-input feature path with:

    client-local raw input
    -> topology-aware local encoder view
    -> frozen CNN / MLPMixer backbone
    -> client embedding
    -> server-side topology-aware fusion feature
    -> topology-regularized affine head

The goal is not to redesign the original paper from scratch. The goal is to
preserve CNN / MLPMixer while making the input interface and fusion path more
industrial-federated and still compatible with the existing affine-head repair
stack.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from utils import NewDataset
from utils.fed_client_runtime import compute_backbone_features
from utils.fed_partial_input import infer_bus_axis
from utils.fed_bus_client import bus_groups_to_string, parse_bus_groups
from utils.topology import shortest_path_distance
from utils.fed_cache_utils import load_topology_fusion_cache, save_topology_fusion_cache


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class TopologyLocalFusionContext:
    dataset_train_affine: NewDataset
    dataset_test_affine: NewDataset
    bus_groups: List[np.ndarray]
    bus_groups_string: str
    bus_axis: int
    client_graph: np.ndarray
    client_graph_smoothed: np.ndarray
    encoder_upload_summary: pd.DataFrame
    client_embedding_summary: pd.DataFrame
    fusion_feature_summary: pd.DataFrame


def make_topology_balanced_bus_groups(
    A_grid: np.ndarray,
    num_clients: int,
) -> List[np.ndarray]:
    """
    Build topology-aware bus groups by nearest-seed assignment on graph distance,
    with a soft balance target.
    """
    A_grid = np.asarray(A_grid, dtype=float)
    num_bus = int(A_grid.shape[0])
    num_clients = int(max(1, num_clients))

    if num_clients >= num_bus:
        return [np.asarray([i], dtype=int) for i in range(num_bus)]

    dist = shortest_path_distance(A_grid)
    finite_dist = np.where(np.isfinite(dist), dist, 1e6)

    seeds = [0]
    while len(seeds) < num_clients:
        best_bus = None
        best_score = -np.inf
        for bus in range(num_bus):
            if bus in seeds:
                continue
            score = float(np.min(finite_dist[bus, seeds]))
            if score > best_score:
                best_bus = bus
                best_score = score
        seeds.append(int(best_bus))

    base = num_bus // num_clients
    remainder = num_bus % num_clients
    capacities = [base + (1 if i < remainder else 0) for i in range(num_clients)]

    assignments = {cid: [seed] for cid, seed in enumerate(seeds)}
    assigned = set(seeds)

    remaining = [bus for bus in range(num_bus) if bus not in assigned]
    scored_remaining = []
    for bus in remaining:
        sorted_clients = sorted(
            range(num_clients),
            key=lambda cid: (finite_dist[bus, seeds[cid]], cid),
        )
        scored_remaining.append((bus, sorted_clients))

    # Harder buses first: those whose best and second-best seeds are far apart.
    scored_remaining.sort(
        key=lambda item: (
            finite_dist[item[0], seeds[item[1][1]]] - finite_dist[item[0], seeds[item[1][0]]]
            if len(item[1]) > 1
            else 1e6
        ),
        reverse=True,
    )

    for bus, order in scored_remaining:
        chosen = None
        for cid in order:
            if len(assignments[cid]) < capacities[cid]:
                chosen = cid
                break
        if chosen is None:
            chosen = int(np.argmin([len(assignments[cid]) for cid in range(num_clients)]))
        assignments[chosen].append(int(bus))

    groups = []
    for cid in range(num_clients):
        group = np.asarray(sorted(assignments[cid]), dtype=int)
        groups.append(group)
    return groups


def resolve_bus_groups_with_topology(
    A_grid: np.ndarray,
    num_clients: int,
    bus_groups_override: Optional[str] = None,
    partition_mode: str = "topology",
) -> List[np.ndarray]:
    if bus_groups_override is not None and str(bus_groups_override).strip() != "":
        return parse_bus_groups(
            bus_groups=bus_groups_override,
            num_bus=A_grid.shape[0],
            num_clients=num_clients,
        )

    if str(partition_mode).lower() in ["topology", "graph", "adjacency"]:
        return make_topology_balanced_bus_groups(A_grid=A_grid, num_clients=num_clients)

    return parse_bus_groups(
        bus_groups=None,
        num_bus=A_grid.shape[0],
        num_clients=num_clients,
    )


def build_bus_propagation_matrix(
    num_bus: int,
    local_buses: Sequence[int],
    dist: np.ndarray,
    tau: float = 1.0,
    self_weight: float = 3.0,
    mode: str = "topology",
) -> np.ndarray:
    """
    Map local bus observations to a full-bus proxy view.

    Each global bus is reconstructed as a topology-weighted combination of the
    locally observed buses. Owned buses are strongly anchored to themselves.
    """
    local_buses = np.asarray(local_buses, dtype=int).reshape(-1)
    mode = str(mode).lower()

    if mode in ["local_only", "local", "identity", "zero_fill_local"]:
        W = np.zeros((int(num_bus), int(len(local_buses))), dtype=np.float64)
        local_pos = {int(bus): idx for idx, bus in enumerate(local_buses)}
        for global_bus, col in local_pos.items():
            W[int(global_bus), int(col)] = 1.0
        return W

    W = np.zeros((int(num_bus), int(len(local_buses))), dtype=np.float64)

    for j in range(int(num_bus)):
        row = []
        for local_col, local_bus in enumerate(local_buses):
            d = dist[j, local_bus]
            if not np.isfinite(d):
                value = 0.0
            else:
                value = float(np.exp(-float(d) / max(float(tau), 1e-6)))
            if int(j) == int(local_bus):
                value *= float(self_weight)
            row.append(value)
        row = np.asarray(row, dtype=np.float64)
        if np.sum(row) <= 0:
            row[:] = 1.0 / max(len(local_buses), 1)
        else:
            row = row / np.sum(row)
        W[j, :] = row

    return W


def _move_bus_axis_to_last(x: np.ndarray, bus_axis: int) -> np.ndarray:
    return np.moveaxis(x, int(bus_axis), -2)


def _move_bus_axis_from_last(x: np.ndarray, bus_axis: int) -> np.ndarray:
    return np.moveaxis(x, -2, int(bus_axis))


def reconstruct_full_input_from_local(
    raw_feature: np.ndarray,
    bus_indices: Sequence[int],
    propagation: np.ndarray,
    bus_axis: int,
) -> np.ndarray:
    """
    Reconstruct a full-bus input tensor from local-only raw inputs.

    Steps:
        1. slice local buses from raw tensor
        2. move bus axis next to feature axis
        3. apply topology propagation over the bus dimension
        4. restore original axis order
    """
    x = _to_numpy(raw_feature).astype(np.float32, copy=False)
    local_buses = np.asarray(bus_indices, dtype=int)

    x_local = np.take(x, indices=local_buses, axis=int(bus_axis))
    x_last = _move_bus_axis_to_last(x_local, int(bus_axis))
    original_shape = x_last.shape
    x_flat = x_last.reshape(-1, original_shape[-2], original_shape[-1])

    propagated = np.einsum("jk,bkf->bjf", propagation.astype(np.float32), x_flat)
    propagated = propagated.reshape(*original_shape[:-2], propagation.shape[0], original_shape[-1])

    return _move_bus_axis_from_last(propagated, int(bus_axis))


def fill_nonlocal_with_mean(
    raw_feature: np.ndarray,
    bus_indices: Sequence[int],
    output_dim: int,
    bus_axis: int,
) -> np.ndarray:
    """
    Keep local bus values and fill non-local buses with per-bus mean values
    estimated from the current split.
    """
    x = _to_numpy(raw_feature).astype(np.float32, copy=False)
    local_buses = np.asarray(bus_indices, dtype=int).reshape(-1)
    out = np.array(x, copy=True)

    x_last = _move_bus_axis_to_last(x, int(bus_axis))
    bus_mean = np.mean(x_last, axis=tuple(range(x_last.ndim - 2)), keepdims=True)
    bus_mean = np.broadcast_to(bus_mean, x_last.shape)
    out_last = _move_bus_axis_to_last(out, int(bus_axis))

    mask = np.zeros((int(output_dim),), dtype=bool)
    mask[local_buses] = True
    out_last[..., ~mask, :] = bus_mean[..., ~mask, :]
    return _move_bus_axis_from_last(out_last, int(bus_axis))


def build_client_graph(bus_groups: Sequence[np.ndarray], dist: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """
    Client graph derived from minimum inter-group topology distance.
    """
    groups = [np.asarray(g, dtype=int).reshape(-1) for g in bus_groups]
    K = len(groups)
    G = np.eye(K, dtype=np.float64)

    for i in range(K):
        for j in range(i + 1, K):
            d = np.min(dist[np.ix_(groups[i], groups[j])])
            if np.isfinite(d):
                w = float(np.exp(-float(d) / max(float(tau), 1e-6)))
            else:
                w = 0.0
            G[i, j] = w
            G[j, i] = w

    row_sum = G.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    return G / row_sum


def smooth_client_embeddings(
    embeddings: np.ndarray,
    client_graph: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend each client embedding with topology-neighbor embeddings.
    """
    E = np.asarray(embeddings, dtype=np.float64)  # [N, K, D]
    neighbor = np.einsum("ij,bjd->bid", client_graph, E)
    return (1.0 - float(alpha)) * E + float(alpha) * neighbor


def compute_topology_local_embeddings(
    core_model,
    raw_feature,
    bus_groups: Sequence[np.ndarray],
    A_grid: np.ndarray,
    tau: float = 1.0,
    self_weight: float = 3.0,
    encoder_topology_mode: str = "topology",
    batch_size: int = 128,
    device: str = "cpu",
    bus_axis: Optional[int] = None,
) -> Tuple[np.ndarray, int, pd.DataFrame]:
    """
    For each client, reconstruct a topology-aware full-bus proxy input from
    local-only raw input, then run the frozen backbone to get a client embedding.
    """
    raw_np = _to_numpy(raw_feature)
    output_dim = int(A_grid.shape[0])
    if bus_axis is None:
        bus_axis = infer_bus_axis(raw_np, output_dim)
    if bus_axis is None:
        raise ValueError("Could not infer bus axis for topology-local encoder path.")

    dist = shortest_path_distance(A_grid)
    per_client_embeddings = []
    rows = []

    for cid, buses in enumerate(bus_groups):
        buses = np.asarray(buses, dtype=int)
        mode = str(encoder_topology_mode).lower()
        if mode == "mean_fill_local":
            propagation = np.zeros((int(output_dim), int(len(buses))), dtype=np.float64)
            reconstructed = fill_nonlocal_with_mean(
                raw_feature=raw_np,
                bus_indices=buses,
                output_dim=output_dim,
                bus_axis=bus_axis,
            )
        else:
            propagation = build_bus_propagation_matrix(
                num_bus=output_dim,
                local_buses=buses,
                dist=dist,
                tau=tau,
                self_weight=self_weight,
                mode=encoder_topology_mode,
            )
            reconstructed = reconstruct_full_input_from_local(
                raw_feature=raw_np,
                bus_indices=buses,
                propagation=propagation,
                bus_axis=bus_axis,
            )
        emb = compute_backbone_features(
            core_model=core_model,
            raw_feature=reconstructed,
            batch_size=batch_size,
            device=device,
        )
        per_client_embeddings.append(emb)

        rows.append(
            {
                "client_id": int(cid),
                "bus_indices": ",".join(str(int(x)) for x in buses),
                "num_buses": int(len(buses)),
                "bus_axis": int(bus_axis),
                "upload_raw_elements": int(np.take(raw_np, buses, axis=bus_axis).size),
                "reconstructed_input_elements": int(reconstructed.size),
                "embedding_dim": int(emb.shape[1]),
                "embedding_norm": float(np.linalg.norm(emb)),
                "propagation_density": float(np.mean(propagation > 1e-8)) if propagation.size else 0.0,
                "encoder_topology_mode": str(encoder_topology_mode),
            }
        )

    stacked = np.stack(per_client_embeddings, axis=1)  # [N, K, D]
    return stacked, int(bus_axis), pd.DataFrame(rows)


def build_topology_local_fusion_datasets(
    dataset_sensitive,
    dataset_test,
    core_model,
    A_grid: np.ndarray,
    num_bus_clients: int,
    bus_groups_override: Optional[str] = None,
    partition_mode: str = "topology",
    tau: float = 1.0,
    encoder_self_weight: float = 3.0,
    encoder_topology_mode: str = "topology",
    topology_fusion_enabled: bool = True,
    fusion_alpha: float = 0.5,
    fusion_feature_layout: str = "concat_residual",
    use_smoothing_residual: bool = True,
    normalize_fusion_feature: bool = True,
    batch_size: int = 128,
    device: str = "cpu",
    cache_dir: Optional[Path] = None,
) -> TopologyLocalFusionContext:
    """
    Build train/test affine datasets from client-local embeddings.

    Feature layout:
        agg_mean:
            mean(active client embedding_k), where active means smoothed when
            topology_fusion_enabled=true, else local
        concat_residual:
            concat(local_embedding_k, active_embedding_k - local_embedding_k)
        concat_local:
            concat(active embedding_k) for k=1..K
        local_mean:
            server consumes only the mean of local embeddings across clients

    Legacy aliases:
        secure_agg_mean       -> agg_mean
        secure_agg_local_mean -> local_mean
    """
    bus_groups = resolve_bus_groups_with_topology(
        A_grid=A_grid,
        num_clients=num_bus_clients,
        bus_groups_override=bus_groups_override,
        partition_mode=partition_mode,
    )

    if cache_dir is not None:
        cached = load_topology_fusion_cache(
            cache_dir=Path(cache_dir),
            target_mean=dataset_sensitive.target_mean,
            target_std=dataset_sensitive.target_std,
            train_target=dataset_sensitive.target,
            test_target=dataset_test.target,
            is_scale=dataset_sensitive.is_scale,
        )
        if cached is not None:
            return TopologyLocalFusionContext(
                dataset_train_affine=cached["dataset_train_affine"],
                dataset_test_affine=cached["dataset_test_affine"],
                bus_groups=cached["bus_groups"],
                bus_groups_string=bus_groups_to_string(cached["bus_groups"]),
                bus_axis=int(cached["bus_axis"]),
                client_graph=cached["client_graph"],
                client_graph_smoothed=cached["client_graph"],
                encoder_upload_summary=cached["encoder_upload_summary"],
                client_embedding_summary=cached["client_embedding_summary"],
                fusion_feature_summary=cached["fusion_feature_summary"],
            )

    dist = shortest_path_distance(A_grid)

    train_embed, bus_axis, upload_train_df = compute_topology_local_embeddings(
        core_model=core_model,
        raw_feature=dataset_sensitive.feature,
        bus_groups=bus_groups,
        A_grid=A_grid,
        tau=tau,
        self_weight=encoder_self_weight,
        encoder_topology_mode=encoder_topology_mode,
        batch_size=batch_size,
        device=device,
        bus_axis=None,
    )
    test_embed, _bus_axis_test, upload_test_df = compute_topology_local_embeddings(
        core_model=core_model,
        raw_feature=dataset_test.feature,
        bus_groups=bus_groups,
        A_grid=A_grid,
        tau=tau,
        self_weight=encoder_self_weight,
        encoder_topology_mode=encoder_topology_mode,
        batch_size=batch_size,
        device=device,
        bus_axis=bus_axis,
    )

    client_graph = build_client_graph(bus_groups=bus_groups, dist=dist, tau=tau)
    train_smooth = smooth_client_embeddings(train_embed, client_graph=client_graph, alpha=fusion_alpha)
    test_smooth = smooth_client_embeddings(test_embed, client_graph=client_graph, alpha=fusion_alpha)
    train_active = train_smooth if topology_fusion_enabled else train_embed
    test_active = test_smooth if topology_fusion_enabled else test_embed

    train_local_flat = train_embed.reshape(train_embed.shape[0], -1)
    test_local_flat = test_embed.reshape(test_embed.shape[0], -1)
    train_active_flat = train_active.reshape(train_active.shape[0], -1)
    test_active_flat = test_active.reshape(test_active.shape[0], -1)
    fusion_feature_layout = str(fusion_feature_layout).lower()
    if fusion_feature_layout == "secure_agg_mean":
        fusion_feature_layout = "agg_mean"
    elif fusion_feature_layout == "secure_agg_local_mean":
        fusion_feature_layout = "local_mean"

    if fusion_feature_layout == "agg_mean":
        train_feature = np.mean(train_active, axis=1)
        test_feature = np.mean(test_active, axis=1)
    elif fusion_feature_layout == "concat_residual":
        if use_smoothing_residual:
            train_smooth_part = (train_active - train_embed).reshape(train_active.shape[0], -1)
            test_smooth_part = (test_active - test_embed).reshape(test_active.shape[0], -1)
        else:
            train_smooth_part = train_active_flat
            test_smooth_part = test_active_flat
        train_feature = np.concatenate([train_local_flat, train_smooth_part], axis=1)
        test_feature = np.concatenate([test_local_flat, test_smooth_part], axis=1)
    elif fusion_feature_layout == "concat_local":
        train_feature = train_active_flat
        test_feature = test_active_flat
    elif fusion_feature_layout == "local_mean":
        train_feature = np.mean(train_embed, axis=1)
        test_feature = np.mean(test_embed, axis=1)
    else:
        raise ValueError(
            "fusion_feature_layout must be one of "
            "agg_mean, local_mean, concat_residual, concat_local."
        )

    feature_mean = np.mean(train_feature, axis=0)
    feature_std = np.std(train_feature, axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    if normalize_fusion_feature:
        train_feature = (train_feature - feature_mean[None, :]) / feature_std[None, :]
        test_feature = (test_feature - feature_mean[None, :]) / feature_std[None, :]

    mean = dataset_sensitive.target_mean
    std = dataset_sensitive.target_std
    dataset_train_affine = NewDataset(
        torch.tensor(train_feature).float(),
        dataset_sensitive.target,
        mean,
        std,
    )
    dataset_test_affine = NewDataset(
        torch.tensor(test_feature).float(),
        dataset_test.target,
        mean,
        std,
    )
    dataset_train_affine.is_scale = dataset_sensitive.is_scale
    dataset_test_affine.is_scale = dataset_test.is_scale

    upload_train_df = upload_train_df.assign(split="train_sensitive")
    upload_test_df = upload_test_df.assign(split="test")
    encoder_upload_summary = pd.concat([upload_train_df, upload_test_df], ignore_index=True)

    client_rows = []
    for cid, buses in enumerate(bus_groups):
        client_rows.append(
            {
                "client_id": int(cid),
                "bus_indices": ",".join(str(int(x)) for x in buses),
                "num_buses": int(len(buses)),
                "client_degree": float(np.sum(client_graph[cid]) - client_graph[cid, cid]),
                "client_graph_row": ",".join(f"{float(v):.6f}" for v in client_graph[cid]),
            }
        )
    client_embedding_summary = pd.DataFrame(client_rows)

    fusion_feature_summary = pd.DataFrame(
        [
            {
                "bus_groups": bus_groups_to_string(bus_groups),
                "bus_axis": int(bus_axis),
                "num_clients": int(len(bus_groups)),
                "local_embedding_dim": int(train_embed.shape[-1]),
                "fusion_feature_dim": int(train_feature.shape[1]),
                "fusion_alpha": float(fusion_alpha),
                "topology_fusion_enabled": bool(topology_fusion_enabled),
                "topology_tau": float(tau),
                "encoder_self_weight": float(encoder_self_weight),
                "encoder_topology_mode": str(encoder_topology_mode),
                "fusion_feature_layout": str(fusion_feature_layout),
                "use_smoothing_residual": bool(use_smoothing_residual),
                "normalize_fusion_feature": bool(normalize_fusion_feature),
                "raw_feature_mean_abs": float(np.mean(np.abs(feature_mean))),
                "raw_feature_std_mean": float(np.mean(feature_std)),
                "train_feature_condition_proxy": float(np.linalg.cond(train_feature.T @ train_feature + 1e-6 * np.eye(train_feature.shape[1]))),
            }
        ]
    )

    ctx = TopologyLocalFusionContext(
        dataset_train_affine=dataset_train_affine,
        dataset_test_affine=dataset_test_affine,
        bus_groups=bus_groups,
        bus_groups_string=bus_groups_to_string(bus_groups),
        bus_axis=int(bus_axis),
        client_graph=client_graph,
        client_graph_smoothed=client_graph,
        encoder_upload_summary=encoder_upload_summary,
        client_embedding_summary=client_embedding_summary,
        fusion_feature_summary=fusion_feature_summary,
    )

    if cache_dir is not None:
        save_topology_fusion_cache(
            cache_dir=Path(cache_dir),
            meta={
                "bus_axis": int(bus_axis),
                "partition_mode": str(partition_mode),
                "tau": float(tau),
                "encoder_self_weight": float(encoder_self_weight),
                "encoder_topology_mode": str(encoder_topology_mode),
                "fusion_feature_layout": str(fusion_feature_layout),
                "fusion_alpha": float(fusion_alpha),
                "use_smoothing_residual": bool(use_smoothing_residual),
                "normalize_fusion_feature": bool(normalize_fusion_feature),
                "num_bus_clients": int(num_bus_clients),
                "bus_groups_string": bus_groups_to_string(bus_groups),
            },
            dataset_train_affine=dataset_train_affine,
            dataset_test_affine=dataset_test_affine,
            bus_groups=bus_groups,
            client_graph=client_graph,
            encoder_upload_summary=encoder_upload_summary,
            client_embedding_summary=client_embedding_summary,
            fusion_feature_summary=fusion_feature_summary,
        )

    return ctx
