from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch

from utils import NewDataset


@dataclass
class LinearClientPartition:
    client_id: int
    bus_indices: np.ndarray
    row_dataset: NewDataset


def default_bus_groups(num_buses: int, num_bus_clients: int = 4) -> List[np.ndarray]:
    return [np.asarray(group, dtype=int) for group in np.array_split(np.arange(num_buses), int(num_bus_clients))]


def flatten_linear_dataset(dataset):
    feature = dataset.feature
    target = dataset.target

    if isinstance(feature, torch.Tensor):
        x = feature.detach().cpu().numpy()
    else:
        x = np.asarray(feature)

    if isinstance(target, torch.Tensor):
        y = target.detach().cpu().numpy()
    else:
        y = np.asarray(target)

    n_sample, n_bus, feature_dim = x.shape
    x_rows = x.reshape(n_sample * n_bus, feature_dim).astype(np.float64)
    y_rows = y.reshape(n_sample * n_bus).astype(np.float64)
    return x_rows, y_rows


def build_linear_client_partitions(dataset, bus_groups: Sequence[np.ndarray]) -> List[LinearClientPartition]:
    feature = dataset.feature
    target = dataset.target

    partitions = []
    for client_id, buses in enumerate(bus_groups):
        buses = np.asarray(buses, dtype=int)
        local_feature = feature[:, buses, :].reshape(-1, feature.shape[-1]).clone()
        local_target = target[:, buses].reshape(-1).clone()
        partitions.append(
            LinearClientPartition(
                client_id=int(client_id),
                bus_indices=buses,
                row_dataset=NewDataset(
                    feature=local_feature,
                    target=local_target,
                    mean=dataset.target_mean,
                    std=dataset.target_std,
                ),
            )
        )
    return partitions


def augmented_design_matrix(row_dataset: NewDataset) -> np.ndarray:
    x = row_dataset.feature.detach().cpu().numpy().astype(np.float64)
    return np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)


def extract_linear_parameter(flat_parameter: np.ndarray):
    theta = np.asarray(flat_parameter, dtype=np.float64).reshape(-1)
    return theta[:-1], float(theta[-1])


def compute_linear_hessian_from_rows(row_dataset: NewDataset, total_row_count: int) -> np.ndarray:
    x_aug = augmented_design_matrix(row_dataset)
    return 2.0 * (x_aug.T @ x_aug) / float(total_row_count)


def aggregate_linear_hessian(partitions: Sequence[LinearClientPartition], total_row_count: int) -> np.ndarray:
    hessian = None
    for part in partitions:
        local_h = compute_linear_hessian_from_rows(part.row_dataset, total_row_count=total_row_count)
        if hessian is None:
            hessian = np.zeros_like(local_h, dtype=np.float64)
        hessian += local_h
    return hessian


def solve_linear_ihvp(hessian: np.ndarray, grad_vec: np.ndarray) -> np.ndarray:
    try:
        return -np.linalg.solve(hessian, grad_vec)
    except np.linalg.LinAlgError:
        return -np.linalg.lstsq(hessian, grad_vec, rcond=None)[0]


def compute_linear_sample_scores_for_buses(
    dataset,
    bus_indices: Sequence[int],
    parameter_vec: np.ndarray,
    M_vec: np.ndarray,
    normalize_by_n: bool = True,
) -> np.ndarray:
    buses = np.asarray(bus_indices, dtype=int)
    weight, bias = extract_linear_parameter(parameter_vec)
    M_weight, M_bias = extract_linear_parameter(M_vec)

    x = dataset.feature[:, buses, :].detach().cpu().numpy().astype(np.float64)
    y = dataset.target[:, buses].detach().cpu().numpy().astype(np.float64)

    pred = np.tensordot(x, weight, axes=([2], [0])) + bias
    err = pred - y

    projected = np.tensordot(x, M_weight, axes=([2], [0])) + M_bias
    score = (2.0 / float(dataset.target.shape[1])) * np.sum(err * projected, axis=1)

    if normalize_by_n:
        score = score / float(len(dataset))

    return score


def compute_linear_sample_scores_global(dataset, parameter_vec: np.ndarray, M_vec: np.ndarray) -> np.ndarray:
    all_buses = np.arange(dataset.target.shape[1], dtype=int)
    return compute_linear_sample_scores_for_buses(
        dataset=dataset,
        bus_indices=all_buses,
        parameter_vec=parameter_vec,
        M_vec=M_vec,
        normalize_by_n=True,
    )


def compute_linear_sample_scores_fed(
    dataset,
    bus_groups: Sequence[np.ndarray],
    parameter_vec: np.ndarray,
    M_vec: np.ndarray,
) -> np.ndarray:
    total = np.zeros((len(dataset),), dtype=np.float64)
    for buses in bus_groups:
        total += compute_linear_sample_scores_for_buses(
            dataset=dataset,
            bus_indices=buses,
            parameter_vec=parameter_vec,
            M_vec=M_vec,
            normalize_by_n=True,
        )
    return total


def linear_partition_summary(partitions: Sequence[LinearClientPartition], total_row_count: int) -> pd.DataFrame:
    rows = []
    for part in partitions:
        local_h = compute_linear_hessian_from_rows(part.row_dataset, total_row_count=total_row_count)
        rows.append(
            {
                "client_id": int(part.client_id),
                "bus_indices": ",".join(str(int(x)) for x in part.bus_indices),
                "num_buses": int(len(part.bus_indices)),
                "num_rows": int(len(part.row_dataset)),
                "feature_dim": int(part.row_dataset.feature.shape[-1]),
                "H_k_norm": float(np.linalg.norm(local_h)),
                "H_k_condition": float(np.linalg.cond(local_h)),
            }
        )
    return pd.DataFrame(rows)


def linear_score_summary(dataset, bus_groups: Sequence[np.ndarray], parameter_vec: np.ndarray, M_vec: np.ndarray) -> pd.DataFrame:
    rows = []
    for client_id, buses in enumerate(bus_groups):
        score = compute_linear_sample_scores_for_buses(
            dataset=dataset,
            bus_indices=buses,
            parameter_vec=parameter_vec,
            M_vec=M_vec,
            normalize_by_n=True,
        )
        rows.append(
            {
                "client_id": int(client_id),
                "bus_indices": ",".join(str(int(x)) for x in buses),
                "num_buses": int(len(buses)),
                "score_min": float(np.min(score)),
                "score_max": float(np.max(score)),
                "score_mean": float(np.mean(score)),
                "score_std": float(np.std(score)),
            }
        )
    return pd.DataFrame(rows)
