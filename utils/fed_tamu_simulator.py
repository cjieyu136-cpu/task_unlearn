"""
utils/fed_tamu_simulator.py

Stage 3C utility module: minimal Fed-TA-MU simulator.

Purpose
-------
This module turns the Stage-3 Fed-VJP audit into a client-sharded simulator.

It preserves the centralized TA-MU mathematics while explicitly separating:

    Client:
        - stores local samples
        - computes local predictions y_hat
        - receives cloud g_y for its samples
        - computes local VJP J_f(theta)^T g_y
        - computes local sample scores grad_i^T M for its samples

    Cloud/Server:
        - receives predictions / target loads for OPF-cost layer
        - computes g_y = d L_cost / d y_hat
        - aggregates client gradients
        - runs the existing repo-style IHVP/reweight pipeline

Scope of this first simulator
-----------------------------
This is a sample-shard federated simulator designed to verify that Fed-TA-MU
preserves centralized TA-MU-cost behavior.

It deliberately does NOT introduce:
    - topology-sparse Hessian
    - bus-client parameter splitting
    - a new unlearn index rule
    - a new OPF objective

It reuses the repository-aligned Stage-2 utilities for IHVP/reweight.
The federated part is the cloud g_y + client VJP and local sample-score
computation.

Repository / history alignment
------------------------------
- OPF/cost layer is from utils.fed_vjp_utils.CloudCostLayer, which mirrors
  original utils.net.Stage_One_Layer / Stage_Two_Layer and func_operation MSE_COST.
- Client VJP uses the same model output space as the topology-affine model.
- Parameter vectors remain compatible with flatten_model/reconstruct_model via
  utils.reweight_utils.get_model_parameter_vector in the runner.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.fed_vjp_utils import (
    CloudCostLayer,
    get_dataset_mean_std,
    model_output_scaled,
    client_vjp_from_gy,
    alignment_metrics,
    centralized_cost_gradient_repo_style,
)
from utils.reweight_utils import compute_sample_scores


# ---------------------------------------------------------------------
# Dataset views and client partition
# ---------------------------------------------------------------------
class ClientDatasetView(torch.utils.data.Dataset):
    """
    Lightweight dataset view over feature/target rows.

    It preserves attributes used by the repo:
        feature, target, is_scale, target_mean, target_std
    """

    def __init__(self, base_dataset: Any, indices: Sequence[int]):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=int).reshape(-1)

        self.feature = base_dataset.feature[self.indices]
        self.target = base_dataset.target[self.indices]

        self.is_scale = getattr(base_dataset, "is_scale", False)
        if hasattr(base_dataset, "target_mean"):
            self.target_mean = base_dataset.target_mean
        if hasattr(base_dataset, "target_std"):
            self.target_std = base_dataset.target_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.feature[idx], self.target[idx]


def make_sample_shards(
    num_samples: int,
    num_clients: int,
    partition: str = "contiguous",
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Create sample shards.

    partition:
        contiguous: split ordered samples into consecutive blocks
        random: random permutation then split
    """
    num_samples = int(num_samples)
    num_clients = int(num_clients)

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    indices = np.arange(num_samples, dtype=int)

    if partition == "random":
        rng = np.random.default_rng(int(seed))
        rng.shuffle(indices)
    elif partition == "contiguous":
        pass
    else:
        raise ValueError("partition must be 'contiguous' or 'random'")

    shards = np.array_split(indices, num_clients)
    return [s.astype(int) for s in shards if len(s) > 0]


def build_client_views(dataset: Any, shards: List[np.ndarray]) -> List[ClientDatasetView]:
    return [ClientDatasetView(dataset, shard) for shard in shards]


# ---------------------------------------------------------------------
# Cloud-side g_y from client predictions
# ---------------------------------------------------------------------
def client_predict_scaled(
    model: torch.nn.Module,
    client_dataset: ClientDatasetView,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Client computes local predictions y_hat in model-native scaled space.
    """
    dev = torch.device(device or "cpu")
    model = model.to(dev)
    model.eval()

    loader = DataLoader(client_dataset, batch_size=int(batch_size), shuffle=False)

    preds = []
    targets = []

    with torch.no_grad():
        for feature, target in loader:
            feature = feature.to(dev)
            pred = model_output_scaled(model, feature)
            preds.append(pred.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())

    return np.concatenate(preds, axis=0).astype(float), np.concatenate(targets, axis=0).astype(float)


def cloud_gy_from_scaled_predictions(
    cfg: Any,
    dataset_like: Any,
    yhat_scaled: np.ndarray,
    target_scaled: np.ndarray,
    batch_size: int = 128,
    device: Optional[str] = None,
    normalize_by_n: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Cloud receives yhat_scaled and target_scaled, then computes:

        g_y_scaled = d L_cost / d yhat_scaled

    No model is used on the cloud side.

    normalize_by_n:
        If None, divide by len(dataset_like).
        For federated aggregation over a global dataset, pass the global N so
        that client gradients sum to the global mean loss gradient.
    """
    dev = torch.device(device or "cpu")
    layer = CloudCostLayer(cfg=cfg, device=dev)

    yhat_scaled = np.asarray(yhat_scaled, dtype=np.float32)
    target_scaled = np.asarray(target_scaled, dtype=np.float32)

    if yhat_scaled.shape != target_scaled.shape:
        raise ValueError(f"Shape mismatch: yhat {yhat_scaled.shape}, target {target_scaled.shape}")

    n_local = yhat_scaled.shape[0]
    n_norm = int(normalize_by_n or n_local)

    gy_chunks = []
    cost_chunks = []

    start = 0
    while start < n_local:
        end = min(start + int(batch_size), n_local)

        yhat = torch.tensor(yhat_scaled[start:end], dtype=torch.float32, device=dev, requires_grad=True)
        target = torch.tensor(target_scaled[start:end], dtype=torch.float32, device=dev)

        mean, std = get_dataset_mean_std(dataset_like, dtype=yhat.dtype, device=dev)
        forecast_physical = yhat * std + mean
        target_physical = target * std + mean

        cost_per_sample, _opf = layer.forward_cost_per_sample(
            forecast_load_physical=forecast_physical,
            target_load_physical=target_physical,
        )

        # Match global mean loss if normalize_by_n is global dataset length.
        loss = torch.sum(cost_per_sample) / float(n_norm)

        if yhat.grad is not None:
            yhat.grad.zero_()

        loss.backward()

        gy_chunks.append(yhat.grad.detach().cpu().numpy())
        cost_chunks.append(cost_per_sample.detach().cpu().numpy())

        start = end

    gy_scaled = np.concatenate(gy_chunks, axis=0).astype(float)
    cost_per_sample = np.concatenate(cost_chunks, axis=0).astype(float)

    return {
        "gy_scaled": gy_scaled,
        "cost_per_sample": cost_per_sample,
        "sample_vjp_score_l1": np.sum(np.abs(gy_scaled), axis=1),
        "sample_vjp_score_l2": np.linalg.norm(gy_scaled, axis=1),
        "info": {
            "num_samples": int(n_local),
            "normalize_by_n": int(n_norm),
            "mean_abs_gy": float(np.mean(np.abs(gy_scaled))),
            "mean_cost": float(np.mean(cost_per_sample)),
        },
    }


# ---------------------------------------------------------------------
# Federated VJP gradient
# ---------------------------------------------------------------------
def federated_cost_gradient_vjp(
    cfg: Any,
    model: torch.nn.Module,
    dataset: Any,
    num_clients: int = 4,
    partition: str = "contiguous",
    seed: int = 42,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Simulate Fed-VJP gradient computation over sample-sharded clients.

    Returns:
        aggregated_grad
        per_client_info
        gy_full in original sample order
    """
    n = len(dataset)
    shards = make_sample_shards(n, num_clients, partition=partition, seed=seed)
    clients = build_client_views(dataset, shards)

    grad_sum = None
    gy_full = np.zeros((n, dataset.target.shape[1]), dtype=float)
    cost_full = np.zeros(n, dtype=float)

    per_client = []

    for cid, client_ds in enumerate(clients):
        yhat_scaled, target_scaled = client_predict_scaled(
            model=model,
            client_dataset=client_ds,
            batch_size=batch_size,
            device=device,
        )

        cloud = cloud_gy_from_scaled_predictions(
            cfg=cfg,
            dataset_like=client_ds,
            yhat_scaled=yhat_scaled,
            target_scaled=target_scaled,
            batch_size=batch_size,
            device=device,
            normalize_by_n=n,
        )

        grad_client = client_vjp_from_gy(
            model=model,
            dataset=client_ds,
            gy_scaled=cloud["gy_scaled"],
            batch_size=batch_size,
            device=device,
        )

        if grad_sum is None:
            grad_sum = grad_client.copy()
        else:
            grad_sum += grad_client

        gy_full[client_ds.indices] = cloud["gy_scaled"]
        cost_full[client_ds.indices] = cloud["cost_per_sample"]

        per_client.append(
            {
                "client_id": int(cid),
                "num_samples": int(len(client_ds)),
                "grad_norm": float(np.linalg.norm(grad_client)),
                "mean_abs_gy": cloud["info"]["mean_abs_gy"],
                "mean_cost": cloud["info"]["mean_cost"],
                "index_min": int(np.min(client_ds.indices)) if len(client_ds) else -1,
                "index_max": int(np.max(client_ds.indices)) if len(client_ds) else -1,
            }
        )

    if grad_sum is None:
        grad_sum = np.array([], dtype=float)

    return {
        "grad_fed": grad_sum.astype(float),
        "gy_full": gy_full,
        "cost_per_sample": cost_full,
        "client_shards": shards,
        "per_client": per_client,
        "info": {
            "num_clients": int(len(clients)),
            "partition": partition,
            "seed": int(seed),
            "num_samples": int(n),
            "grad_norm": float(np.linalg.norm(grad_sum)),
        },
    }


# ---------------------------------------------------------------------
# Federated sample-score computation
# ---------------------------------------------------------------------
def federated_sample_scores_by_clients(
    cfg: Any,
    model_train: torch.nn.Module,
    dataset_score: Any,
    loader_hessian: DataLoader,
    M_vec: np.ndarray,
    num_clients: int = 4,
    partition: str = "contiguous",
    seed: int = 42,
    batch_size: int = 128,
    train_loss: str = "mse",
    normalize_by_n: bool = True,
) -> Dict[str, Any]:
    """
    Client-local computation of sample scores grad_i^T M.

    Each client computes scores for its local dataset view. The server scatters
    them back to global sample order.

    This reuses the repository-aligned compute_sample_scores utility, avoiding
    a hand-written per-sample loss implementation.
    """
    n = len(dataset_score)
    shards = make_sample_shards(n, num_clients, partition=partition, seed=seed)
    clients = build_client_views(dataset_score, shards)

    scores_full = np.zeros(n, dtype=float)
    per_client = []

    for cid, client_ds in enumerate(clients):
        loader_client = DataLoader(client_ds, batch_size=int(batch_size), shuffle=False)

        # IMPORTANT:
        # The original repo-style score is:
        #     score_i = grad_i^T M
        #     scores = scores / len(global_score_dataset)
        #
        # compute_sample_scores(..., normalize_by_n=True) divides by
        # len(dataset_score). If called on a client view, that would divide by
        # the local client size and inflate scores by roughly K clients.
        #
        # Therefore each client computes the unnormalized local scores, and the
        # server applies the global normalization by len(dataset_score).
        scores_local_raw, info = compute_sample_scores(
            cfg=cfg,
            model_train=model_train,
            loader_hessian=loader_hessian,
            loader_score=loader_client,
            dataset_score=client_ds,
            M_vec=M_vec,
            train_loss=train_loss,
            normalize_by_n=False,
        )

        scores_local_raw = np.asarray(scores_local_raw, dtype=float).reshape(-1)
        if len(scores_local_raw) != len(client_ds):
            raise RuntimeError(
                f"Client {cid} score length mismatch: {len(scores_local_raw)} vs {len(client_ds)}"
            )

        if normalize_by_n:
            scores_local = scores_local_raw / float(len(dataset_score))
        else:
            scores_local = scores_local_raw

        scores_full[client_ds.indices] = scores_local

        per_client.append(
            {
                "client_id": int(cid),
                "num_samples": int(len(client_ds)),
                "score_min": float(np.min(scores_local)),
                "score_max": float(np.max(scores_local)),
                "score_mean": float(np.mean(scores_local)),
                "score_std": float(np.std(scores_local)),
                "score_raw_min": float(np.min(scores_local_raw)),
                "score_raw_max": float(np.max(scores_local_raw)),
                "score_raw_mean": float(np.mean(scores_local_raw)),
                "global_score_normalization_n": int(len(dataset_score)) if normalize_by_n else None,
            }
        )

    return {
        "scores": scores_full,
        "client_shards": shards,
        "per_client": per_client,
        "info": {
            "num_clients": int(len(clients)),
            "partition": partition,
            "seed": int(seed),
            "num_samples": int(n),
            "score_min": float(np.min(scores_full)),
            "score_max": float(np.max(scores_full)),
            "score_mean": float(np.mean(scores_full)),
            "score_std": float(np.std(scores_full)),
        },
    }


# ---------------------------------------------------------------------
# Full audit convenience
# ---------------------------------------------------------------------
def compare_federated_to_central_gradient(
    cfg: Any,
    model: torch.nn.Module,
    dataset: Any,
    num_clients: int = 4,
    partition: str = "contiguous",
    seed: int = 42,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare sample-sharded Fed-VJP gradient to centralized repo-style gradient.
    """
    fed = federated_cost_gradient_vjp(
        cfg=cfg,
        model=model,
        dataset=dataset,
        num_clients=num_clients,
        partition=partition,
        seed=seed,
        batch_size=batch_size,
        device=device,
    )

    central = centralized_cost_gradient_repo_style(
        cfg=cfg,
        model=model,
        dataset=dataset,
        batch_size=batch_size,
    )

    metrics = alignment_metrics(central, fed["grad_fed"])

    return {
        "grad_centralized": central,
        "grad_fed": fed["grad_fed"],
        "gy_full": fed["gy_full"],
        "cost_per_sample": fed["cost_per_sample"],
        "alignment": metrics,
        "fed_info": fed["info"],
        "per_client": fed["per_client"],
    }
